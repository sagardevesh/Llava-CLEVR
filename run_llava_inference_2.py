#!/usr/bin/env python
import argparse
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def normalize_answer(ans):
    """Lowercase and remove punctuation for a simple match."""
    ans = ans.lower().strip()
    # Remove punctuation using regex
    ans = re.sub(r'[^\w\s]', '', ans)
    return ans

def run_inference(model, processor, image_path, question_text):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
    
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_text = processor.decode(output[0][2:], skip_special_tokens=True).strip()
    return generated_text


def main(args):
    # Load test questions JSON file.
    with open(args.questions_file, 'r') as f:
        data = json.load(f)
    
    # Check the structure of the CLEVR questions file
    if "questions" in data:
        questions = data["questions"]
    else:
        questions = data  # Assume it's already a list

    if not questions:
        print("No questions found in the JSON file.")
        return

    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the LLava model.
    model = LlavaForConditionalGeneration.from_pretrained(
        "/fs01/model-weights/llava-1.5-13b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    
    processor = AutoProcessor.from_pretrained("/fs01/model-weights/llava-1.5-13b-hf")
    
    # Explicitly set the patch_size - this is crucial for LLaVA 1.5
    processor.patch_size = 14
    
    # Ensure the vision config has the right image size
    if hasattr(model.config, "vision_config"):
        print(f"Model vision config: image_size={model.config.vision_config.image_size}, patch_size={model.config.vision_config.patch_size}")
        # Ensure processor patch size matches model config
        if hasattr(model.config.vision_config, "patch_size"):
            processor.patch_size = model.config.vision_config.patch_size
            print(f"Set processor patch_size to {processor.patch_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    correct_count = 0
    total_count = 0

    # Process a small batch first to test
    test_limit = args.test_limit if hasattr(args, 'test_limit') else None
    
    for idx, q in enumerate(questions):
        if test_limit and idx >= test_limit:
            break
            
        # CLEVR JSON structure handling
        if "image_filename" in q:
            image_file = q["image_filename"]
        elif "image_index" in q:
            # CLEVR format often uses indices instead of filenames
            image_index = q["image_index"]
            image_file = f"CLEVR_val_{image_index:06d}.png"
        else:
            print(f"Cannot determine image filename from question: {q}")
            continue
            
        question_text = q.get("question")
        ground_truth = q.get("answer", "")  # Default to empty if not found

        if not question_text:
            print(f"Skipping a question due to missing question field: {q}")
            continue

        image_path = os.path.join(args.input_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        print(f"Processing image: {image_path}")
        try:
            generated_answer = run_inference(model, processor, image_path, question_text)
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
            
        if generated_answer is None:
            continue

        total_count += 1
        
        # Only evaluate if ground truth is available
        is_correct = False
        if ground_truth:
            # Normalize answers for a simple exact match evaluation.
            norm_gen = normalize_answer(generated_answer)
            norm_gt = normalize_answer(ground_truth)
            is_correct = norm_gen == norm_gt
            if is_correct:
                correct_count += 1

        results.append({
            "image_file": image_file,
            "question": question_text,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "correct": is_correct
        })
        print(f"Question: {question_text}")
        print(f"Ground Truth: {ground_truth} | Generated: {generated_answer}")
        if ground_truth:
            print(f"Correct: {is_correct}")
        print(f"{'-'*40}")

    # Compute and print overall accuracy.
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Total questions processed: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}")

    # Save detailed results.
    output_file = os.path.join(args.output_dir, "inference_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLava inference on CLEVR test set and evaluate accuracy.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CLEVR test images.")
    parser.add_argument("--questions_file", type=str, required=True, help="Path to CLEVR_test_questions.json file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference results and evaluation.")
    parser.add_argument("--test_limit", type=int, help="Limit the number of test examples (for debugging)")
    args = parser.parse_args()
    
    main(args)