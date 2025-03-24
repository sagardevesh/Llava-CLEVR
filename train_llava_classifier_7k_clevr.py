#!/usr/bin/env python3

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# 1) BitsAndBytes for 4-bit quant
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          LlavaForConditionalGeneration)

###############################################################################
# CLEVR answer set
###############################################################################
CLEVR_ANSWERS = [
    "yes", "no",
    "0","1","2","3","4","5","6","7","8","9","10",
    "red","blue","green","purple","cyan","yellow","brown","gray",
    "cube","sphere","cylinder",
    "large","small",
    "metal","rubber"
]
ANSWER_TO_LABEL = {ans: i for i, ans in enumerate(CLEVR_ANSWERS)}

###############################################################################
# 2) BitsAndBytesConfig for 4-bit
###############################################################################
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # you can also try torch.bfloat16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",            # or "fp4"
)

###############################################################################
# Dataset class that picks subfolder train/val based on filename
###############################################################################
class ClevrClassifierDataset(Dataset):
    def __init__(self, images_dir, questions_json, processor, max_images=None):
        """
        images_dir: top-level folder with 'train' and 'val' subfolders
        questions_json: path to e.g. CLEVR_train_questions.json or CLEVR_val_questions.json
        processor: LLaVA processor
        max_images: optional cap
        """
        self.images_dir = images_dir
        self.processor = processor
        self.data = []

        # load JSON
        with open(questions_json, 'r') as f:
            questions_data = json.load(f)
        if "questions" in questions_data:
            questions_list = questions_data["questions"]
        else:
            questions_list = questions_data

        image_to_qas = defaultdict(list)
        for q in questions_list:
            image_fname = q["image_filename"]  # e.g. "CLEVR_train_000123.png" or "CLEVR_val_000456.png"
            answer_str = q["answer"]
            if answer_str not in ANSWER_TO_LABEL:
                continue
            image_to_qas[image_fname].append(q)

        all_image_filenames = list(image_to_qas.keys())
        if max_images is not None and max_images < len(all_image_filenames):
            sampled = random.sample(all_image_filenames, max_images)
        else:
            sampled = all_image_filenames

        for img_fname in sampled:
            for q in image_to_qas[img_fname]:
                label = ANSWER_TO_LABEL[q["answer"]]
                # Decide subfolder: "train" if "CLEVR_train_" in filename, else "val"
                if "CLEVR_train_" in img_fname:
                    subfolder = "train"
                else:
                    subfolder = "val"
                img_path = os.path.join(self.images_dir, subfolder, img_fname)

                self.data.append({
                    "image_path": img_path,
                    "question": q["question"],
                    "label": label
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


###############################################################################
# Simple linear classifier
###############################################################################
class SimpleClassifier(nn.Module):
    def __init__(self, embed_dim=4096, num_classes=len(CLEVR_ANSWERS)):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

###############################################################################
# Collate
###############################################################################
def collate_fn(batch_items):
    # minimal
    images = []
    texts = []
    labels = []

    for item in batch_items:
        img_path = item["image_path"]
        question = item["question"]
        label = item["label"]

        # load image
        img = Image.open(img_path).convert("RGB")
        images.append(img)

        texts.append(question)  # keep it short
        labels.append(label)

    # process
    proc_out = processor(
        images=images,
        text=texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=32
    )

    batch = {
        "images": proc_out["pixel_values"],
        "input_ids": proc_out["input_ids"],
        "attention_mask": proc_out["attention_mask"],
        "labels": torch.LongTensor(labels),
    }
    return batch


###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--train_questions", type=str, required=True)
    parser.add_argument("--val_questions", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_train_images", type=int, default=7000)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=4096)
    parser.add_argument("--save_dir", type=str, default="classifier_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator()
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes
    is_main_process = accelerator.is_main_process

    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)

    # load LLaVA in 4-bit
    print("Loading LLaVA in 4-bit quantization...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,  # 4-bit!
        device_map=None,                  # let accelerate handle distribution
    )
    model.eval()

    # freeze
    for param in model.parameters():
        param.requires_grad = False

    global processor
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    processor.patch_size = 14  # for LLaVA 1.5

    # more seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Datasets
    train_dataset = ClevrClassifierDataset(
        images_dir=args.images_dir,
        questions_json=args.train_questions,
        processor=processor,
        max_images=args.max_train_images
    )
    val_dataset = ClevrClassifierDataset(
        images_dir=args.images_dir,
        questions_json=args.val_questions,
        processor=processor,
        max_images=args.max_val_images
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset,   num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        collate_fn=collate_fn
    )

    # prepare
    model = accelerator.prepare(model)
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    # extract embeddings train
    train_embs, train_lbls = extract_embeddings_distributed(accelerator, model, train_loader)
    # extract embeddings val
    val_embs, val_lbls = extract_embeddings_distributed(accelerator, model, val_loader)

    # rank 0: train classifier
    if is_main_process:
        classifier = SimpleClassifier(embed_dim=args.embedding_dim, num_classes=len(CLEVR_ANSWERS))
        classifier = classifier.to(accelerator.device)

        train_classifier(classifier, train_embs, train_lbls, val_embs, val_lbls, args.epochs, args.lr, accelerator.device)

        val_acc = evaluate_classifier(classifier, val_embs, val_lbls, accelerator.device)
        print(f"Final val accuracy: {val_acc:.4f}")

        out_path = os.path.join(args.save_dir, "classifier_head_4bit.pt")
        torch.save(classifier.state_dict(), out_path)
        print(f"Classifier saved to {out_path}")
    else:
        pass


###############################################################################
# Embedding extraction
###############################################################################
@torch.no_grad()
def extract_embeddings_distributed(accelerator, model, dataloader):
    model.eval()
    local_embs = []
    local_labels = []

    for batch in dataloader:
        images = batch["images"].to(accelerator.device)  # it's already 4-bit model; no need extra dtype
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        labels = batch["labels"]

        outputs = model(
            vision_inputs=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
        emb = last_hidden[:, -1, :]              # (B, hidden_dim)

        local_embs.append(emb)
        local_labels.append(labels)

    local_embs = torch.cat(local_embs, dim=0)
    local_labels = torch.cat(local_labels, dim=0)

    gathered_embs = accelerator.gather(local_embs)
    gathered_labels = accelerator.gather(local_labels)

    if accelerator.is_main_process:
        return gathered_embs, gathered_labels
    else:
        return None, None


###############################################################################
# Classifier training
###############################################################################
def train_classifier(classifier, train_emb, train_lbl, val_emb, val_lbl, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)

    train_emb = train_emb.to(device)
    train_lbl = train_lbl.to(device)
    val_emb   = val_emb.to(device)
    val_lbl   = val_lbl.to(device)

    for ep in range(epochs):
        classifier.train()

        # Forward + backward on the train embeddings
        logits = classifier(train_emb)
        loss = criterion(logits, train_lbl)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Now measure both train_acc and val_acc
        train_acc = evaluate_classifier(classifier, train_emb, train_lbl, device)
        val_acc   = evaluate_classifier(classifier, val_emb, val_lbl, device)

        print(f"Epoch {ep+1}/{epochs}, loss={loss.item():.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")


@torch.no_grad()
def evaluate_classifier(classifier, emb, lbl, device):
    classifier.eval()
    emb = emb.to(device)
    lbl = lbl.to(device)
    logits = classifier(emb)
    preds = logits.argmax(dim=1)
    correct = (preds == lbl).sum().item()
    total = lbl.size(0)
    return correct / total


if __name__ == "__main__":
    main()
