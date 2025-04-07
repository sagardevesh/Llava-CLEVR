import argparse
import json
import os
import random
from collections import defaultdict

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="Path to CLEVR images directory")
    parser.add_argument("--val_questions", type=str, required=True, help="Path to validation questions JSON file")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base LLaVA model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--num_samples", type=int, default=1500, help="Number of validation samples to evaluate")
    parser.add_argument("--output_file", type=str, default="inference_results.json", help="Output file for results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    return parser.parse_args()

def load_model_and_processor(args):
    print("Loading base model and LoRA adapter...")
    # Load base model (without quantization for inference)
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    processor.patch_size = 14  # Match the value used during training
    processor.num_additional_image_tokens = 0  # Match the value used during training
    
    return model, processor

def prepare_val_data(args, processor):
    print("Preparing validation data...")
    val_data = []
    
    with open(args.val_questions, "r") as f:
        questions_data = json.load(f)
    
    questions_list = questions_data["questions"] if "questions" in questions_data else questions_data
    
    # Group questions by image
    image_to_qas = defaultdict(list)
    for q in questions_list:
        fname = q["image_filename"]
        image_to_qas[fname].append(q)
    
    # Randomly select images
    all_fnames = list(image_to_qas.keys())
    selected_fnames = random.sample(all_fnames, min(args.num_samples, len(all_fnames)))
    
    for fname in selected_fnames:
        for q in image_to_qas[fname]:
            image_path = os.path.join(args.images_dir, "val", fname)
            
            val_data.append({
                "image_path": image_path,
                "question": q["question"],
                "answer": q["answer"],
                "image_filename": fname,
                "question_index": q.get("question_index", -1)
            })
    
    # Limit to requested number of samples
    if len(val_data) > args.num_samples:
        val_data = random.sample(val_data, args.num_samples)
    
    print(f"Prepared {len(val_data)} validation samples")
    return val_data

def normalize_answer(answer):
    """Normalize answer for better comparison."""
    # Convert to lowercase
    answer = answer.lower()
    # Remove punctuation and extra whitespace
    import re
    answer = re.sub(r'[^\w\s]', '', answer).strip()
    # Handle common number words
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12'
    }
    # Replace number words with digits
    for word, digit in number_map.items():
        if answer == word:
            answer = digit
            break
    return answer

def process_batch(batch, model, processor, device):
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    questions = [f"USER: <image>\nQuestion: {item['question']}\nASSISTANT:" for item in batch]
    
    inputs = processor(
        text=questions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False
        )
    
    generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
    responses = []
    
    for gen_text in generated_texts:
        # Extract answer from the generated text
        response = extract_answer(gen_text)
        responses.append(response)
    
    return responses

def extract_answer(gen_text):
    """Extract the answer part from generated text."""
    lower_text = gen_text.lower()
    
    # Look for "the answer is" pattern
    idx = lower_text.find("the answer is")
    if idx >= 0:
        after_str = lower_text[idx+len("the answer is"):].strip()
        # Extract the first word/phrase as the answer
        import re

        # Match until punctuation or common ending words
        match = re.search(r'([^.,;!?]+)', after_str)
        return match.group(1).strip() if match else ""
    
    # If "the answer is" pattern not found, use a more general approach
    response_part = lower_text.split("assistant:")[-1].strip()
    # Return the first meaningful word/phrase
    words = response_part.split()
    if words:
        # Return up to 3 words as the answer
        return " ".join(words[:3]).strip()
    
    return ""

def run_inference(args):
    model, processor = load_model_and_processor(args)
    val_data = prepare_val_data(args, processor)
    device = model.device
    
    results = []
    correct = 0
    total = 0
    
    # Process in batches
    for i in tqdm(range(0, len(val_data), args.batch_size)):
        batch = val_data[i:i+args.batch_size]
        predicted_answers = process_batch(batch, model, processor, device)
        
        for j, item in enumerate(batch):
            # Normalize predicted and ground truth answers
            norm_predicted = normalize_answer(predicted_answers[j])
            norm_ground_truth = normalize_answer(item["answer"])
            
            # Check for match
            is_correct = norm_predicted == norm_ground_truth
            
            result = {
                "image_filename": item["image_filename"],
                "question_index": item["question_index"],
                "question": item["question"],
                "ground_truth": item["answer"],
                "predicted_answer": predicted_answers[j],
                "is_correct": is_correct
            }
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append(result)
    
    # Calculate per-category accuracy if available
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump({
            "overall_accuracy": correct/total if total > 0 else 0,
            "correct_count": correct,
            "total_count": total,
            "results": results
        }, f, indent=2)
    
    # Print overall accuracy
    print(f"Overall accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    random.seed(42)  # For reproducibility
    run_inference(args)