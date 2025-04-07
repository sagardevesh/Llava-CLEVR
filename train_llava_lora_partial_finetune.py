#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict

import accelerate
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
# For LoRA partial finetuning
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# For 8-bit or 4-bit quantization
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          LlavaForConditionalGeneration)

##############################################################################
# 1) LoRA config
##############################################################################
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

##############################################################################
# 2) BitsAndBytesConfig for 4-bit or 8-bit
##############################################################################
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # For 4-bit, use load_in_4bit=True and remove load_in_8bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4"
)

##############################################################################
# 3) Dataset
##############################################################################
class ClevrFinetuneDataset(Dataset):
    """
    Standard supervised training for single-word answers.
    The prompt is built at train time without feeding the answer in,
    so the model must generate the answer itself.
    """
    def __init__(self, images_dir, questions_file, processor, max_images=None, split="train"):
        self.images_dir = images_dir
        self.processor = processor
        self.data = []
        self.split = split

        with open(questions_file, "r") as f:
            d = json.load(f)
        questions_list = d["questions"] if "questions" in d else d

        image_to_qas = defaultdict(list)
        for q in questions_list:
            fname = q["image_filename"]
            image_to_qas[fname].append(q)

        all_fnames = list(image_to_qas.keys())
        used = random.sample(all_fnames, max_images) if max_images is not None and max_images < len(all_fnames) else all_fnames

        for fn in used:
            for q in image_to_qas[fn]:
                subfolder = "train" if "CLEVR_train_" in fn else "val"
                full_path = os.path.join(self.images_dir, subfolder, fn)
                self.data.append({
                    "image_path": full_path,
                    "question": q["question"],
                    "answer": q["answer"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    images = []
    prompts = []
    answers = []

    for item in batch:
        img_path = item["image_path"]
        question = item["question"]
        answer = item["answer"]

        # Load image
        img = Image.open(img_path).convert("RGB")
        images.append(img)

        # Build prompt: the image placeholder will be expanded by the processor.
        text_prompt = f"USER: <image>\nQuestion: {question}\nASSISTANT: "
        text_prompt += f"The answer is {answer}"
        prompts.append(text_prompt)
        answers.append(answer)

    # Increase max_length to ensure all image placeholder tokens are kept.
    proc_out = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    input_ids = proc_out["input_ids"]
    labels = input_ids.clone()

    return {
        "pixel_values": proc_out["pixel_values"],
        "input_ids": input_ids,
        "attention_mask": proc_out["attention_mask"],
        "labels": labels,
        "answers": answers,
    }

def parse_predicted_word(gen_text):
    lower_text = gen_text.lower()
    idx = lower_text.find("the answer is")
    if idx < 0:
        tokens = lower_text.strip().split()
        return tokens[-1] if tokens else ""
    after_str = lower_text[idx+len("the answer is"):].strip()
    first_word = after_str.split()[0] if after_str else ""
    import re
    return re.sub(r"[^a-z0-9]", "", first_word)

##############################################################################
# Evaluation function with proper model unwrapping
##############################################################################
@torch.no_grad()
def evaluate_lora_accuracy(model, val_loader, device):
    if hasattr(model, "module"):
        model_for_eval = model.module
    elif hasattr(model, "_orig_module"):
        model_for_eval = model._orig_module
    elif hasattr(model, "model"):
        model_for_eval = model.model
    else:
        model_for_eval = model
    
    model_for_eval.eval()
    correct = 0
    total = 0

    for batch in val_loader:
        try:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answers"]

            outputs = model_for_eval(
                pixel_values=pixel_values,
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1]
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits[:, -1, :], dim=-1)
            
            if hasattr(model_for_eval, "tokenizer"):
                tokenizer = model_for_eval.tokenizer
            else:
                tokenizer = processor.tokenizer
            
            for i, pred_token_id in enumerate(predictions):
                predicted_word = tokenizer.decode([pred_token_id]).strip().lower()
                gold_answer = answers[i]
                # Fix for handling different answer types (bool, int, etc.)
                if not isinstance(gold_answer, str):
                    gold_answer = str(gold_answer)
                gold_answer = gold_answer.strip().lower()
                
                if predicted_word in gold_answer or gold_answer in predicted_word:
                    correct += 1
                total += 1
                
            print(f"Batch accuracy: {correct}/{total}")
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            print(traceback.format_exc())
            print("Skipping batch evaluation")
            continue

    return correct / total if total > 0 else 0.0

##############################################################################
# Main function
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--train_questions", type=str, required=True)
    parser.add_argument("--val_questions", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_train_images", type=int, default=7000)
    parser.add_argument("--max_val_images", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)  # Changed to 40 epochs
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="lora_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0001,
                        help="Minimum change in validation accuracy to qualify as improvement")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Accelerate version (runtime):", accelerate.__version__)
    print("Accelerate is coming from:", accelerate.__file__)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=False
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes
    is_main_process = accelerator.is_main_process

    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        best_model_dir = f"{args.save_dir}_best"
        os.makedirs(best_model_dir, exist_ok=True)

    print("Loading base LLaVA + LoRA config ...")
    from peft import LoraConfig, TaskType, get_peft_model

    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        device_map=None
    )

    for param in base_model.parameters():
        param.requires_grad = False

    lora_model = get_peft_model(base_model, lora_config)

    global processor
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    processor.patch_size = 14
    # Set additional image tokens to 0 so that the processor inserts exactly (image_size/patch_size)^2 tokens.
    processor.num_additional_image_tokens = 0

    train_dataset = ClevrFinetuneDataset(
        args.images_dir,
        args.train_questions,
        processor,
        max_images=args.max_train_images
    )
    val_dataset = ClevrFinetuneDataset(
        args.images_dir,
        args.val_questions,
        processor,
        max_images=args.max_val_images
    )

    train_sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, rank=local_rank, num_replicas=world_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=2,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    lora_model, optimizer, train_loader, val_loader = accelerator.prepare(
        lora_model, optimizer, train_loader, val_loader
    )

    # Initialize early stopping variables
    best_val_acc = 0.0
    best_epoch = -1
    no_improvement = 0
    
    # Initialize metrics tracking for plotting learning curves
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        lora_model.train()
        train_sampler.set_epoch(epoch)
        
        total_loss = 0
        steps = 0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(lora_model):
                pixel_values = batch["pixel_values"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                outputs = lora_model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                if torch.isnan(loss).item():
                    print(f"WARNING: NaN loss detected at epoch {epoch+1}, step {step}. Skipping update.")
                    continue
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                steps += 1

            if step % 50 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch+1} Step {step}, loss={loss.item():.4f}")

        avg_loss = total_loss / max(steps, 1)
        train_losses.append(avg_loss)
        accelerator.print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

        # Evaluation phase
        val_acc = 0.0
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(lora_model)
            accelerator.print("Running evaluation...")
            try:
                val_acc = evaluate_lora_accuracy(unwrapped_model, val_loader, accelerator.device)
                val_accuracies.append(val_acc)
                accelerator.print(f"Epoch {epoch+1}/{args.epochs}, val_accuracy={val_acc:.4f}")
                
                # Check for improvement
                if val_acc > best_val_acc + args.min_delta:
                    accelerator.print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                    best_val_acc = val_acc
                    best_epoch = epoch
                    no_improvement = 0
                    
                    # Save best model
                    unwrapped_model.save_pretrained(f"{args.save_dir}_best")
                    accelerator.print(f"New best model saved to {args.save_dir}_best")
                else:
                    no_improvement += 1
                    accelerator.print(f"No improvement for {no_improvement} epochs. Best accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
            except Exception as e:
                accelerator.print(f"Evaluation error: {e}")
                accelerator.print("Skipping evaluation for this epoch")
                val_accuracies.append(val_accuracies[-1] if val_accuracies else 0.0)  # Keep the last value if there's an error
        
        # Broadcast validation accuracy to all processes
        val_acc_tensor = torch.tensor([val_acc], device=accelerator.device)
        accelerator.wait_for_everyone()
        gathered_val_acc = accelerator.gather(val_acc_tensor).mean().item()
        
        # Early stopping check
        if no_improvement >= args.patience:
            accelerator.print(f"Early stopping triggered after {epoch+1} epochs. Best accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
            break
        
        accelerator.wait_for_everyone()

    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(lora_model)
        unwrapped_model.save_pretrained(args.save_dir)
        print(f"Final model saved to {args.save_dir}")
        
        # Save training metrics for later analysis
        metrics = {
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "epochs_trained": epoch + 1
        }
        with open(f"{args.save_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Training metrics saved to {args.save_dir}/training_metrics.json")
    
    accelerator.wait_for_everyone()
    
    # Print final summary
    if accelerator.is_main_process:
        if best_epoch >= 0:
            print("\n" + "="*50)
            print(f"Training completed after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")
            print(f"Best model saved to: {args.save_dir}_best")
            print(f"Final model saved to: {args.save_dir}")
            print("="*50)

if __name__ == "__main__":
    main()