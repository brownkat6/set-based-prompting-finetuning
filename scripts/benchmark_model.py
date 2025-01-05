# Standard library imports
import argparse
import datetime
import json
import os
from typing import Dict, List, Tuple

# Third-party imports
import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel

# Records wikitext-103 perplexity for a model at the given model path
# Assumes that the model has a tokenizer in the same directory as the model

def compute_perplexity(model, tokenizer, dataset, max_length=1024, device="cuda"):
    model.eval()
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for i, text in enumerate(tqdm(dataset["text"], desc="Computing perplexity")):
            # Skip empty or whitespace-only texts
            if not text or text.isspace():
                continue
                
            encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = encodings.input_ids.to(device)
            
            # Skip if we only got special tokens (usually shape [1, 1] or [1, 2])
            if input_ids.size(1) <= 2:
                continue
                
            target_ids = input_ids.clone()
            
            outputs = model(input_ids, labels=target_ids)
            batch_loss = outputs.loss.item()
            token_count = target_ids.size(1)
            
            # Debug prints for first few batches
            if i < 5:
                print(f"\nBatch {i}:")
                print(f"Raw loss: {batch_loss}")
                print(f"Token count: {token_count}")
                print(f"Scaled loss: {batch_loss * token_count}")
                print(input_ids)
                print(model)
                
            
            # Check for NaN before accumulating
            if torch.isnan(outputs.loss):
                print(f"\nNaN detected in batch {i}!")
                print(f"Input text length: {len(text)}")
                print(f"Input ids shape: {input_ids.shape}")
                continue
                
            loss = outputs.loss * token_count
            total_loss += loss.item()
            total_length += token_count
            
            # Check running totals periodically
            if i % 100 == 0 and i > 0:
                print(f"\nAfter {i} batches:")
                print(f"Current total loss: {total_loss}")
                print(f"Current total length: {total_length}")
    
    if total_length == 0:
        print("Warning: No valid batches processed!")
        return float('nan')
        
    avg_loss = total_loss / total_length
    if avg_loss > 20:
        print(f"Warning: Very high loss value detected: {avg_loss}")
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Total loss: {total_loss}")
    print(f"Total length: {total_length}")
    print(f"Average loss: {avg_loss}")
    return perplexity.item()

def main():
    print("\n=== Starting Benchmark Process ===")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    args = parser.parse_args()

    # Define benchmarks file path relative to model directory
    model_dir = os.path.dirname(os.path.dirname(args.model_path))  # Go up two levels to get main model dir
    RECORDS_FP = os.path.join(model_dir, "benchmarks.jsonl")

    # Input validation
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    # Ensure CUDA is available if device is cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but cuda device was specified")

    # Create directory for records if it doesn't exist
    os.makedirs(os.path.dirname(RECORDS_FP), exist_ok=True)

    print("\n1. Loading Model and Tokenizer...")
    try:
        # Always load tokenizer from initial_weights
        tokenizer_path = os.path.join(os.path.dirname(args.model_path), "initial_weights")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = args.model_path  # Fallback to model path if initial_weights doesn't exist
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        
        # Load base model and merge LoRA weights if they exist
        print(f"Loading base model from {args.model_path}")
        base_model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto"  # This will handle device placement automatically
        )
        
        # Check for and load LoRA weights if they exist
        adapter_path = os.path.join(args.model_path, "adapter_model.bin")
        if os.path.exists(adapter_path):
            from peft import PeftModel
            print("Found LoRA weights, loading and merging...")
            model = PeftModel.from_pretrained(base_model, args.model_path)
            model = model.merge_and_unload()  # Merge LoRA weights into base model
        else:
            model = base_model
            
        model.to(args.device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")

    print("\n2. Loading WikiText-103 Dataset...")
    try:
        wikitext = load_dataset("wikitext", "wikitext-103-v1", split="test")
    except Exception as e:
        raise RuntimeError(f"Failed to load WikiText dataset: {str(e)}")

    print("\n3. Computing WikiText Perplexity...")
    perplexity = compute_perplexity(model, tokenizer, wikitext, device=args.device)
    print(f"WikiText-103 Perplexity: {perplexity:.2f}")

    print("\n4. Saving Results...")
    results = {
        "model_path": args.model_path,
        "wikitext_perplexity": perplexity,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    try:
        with open(RECORDS_FP, "a") as f:
            f.write(json.dumps(results) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write results to file: {str(e)}")

    print("\n=== Benchmark Process Complete ===\n")

if __name__ == "__main__":
    main()
