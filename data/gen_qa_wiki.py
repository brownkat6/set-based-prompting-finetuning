import json
import os
import glob
import random
from pathlib import Path
from datasets import load_dataset

def process_file(input_path: str, output_path: str, wiki_texts):
    """Process a single input file and add WikiText entries"""
    
    # Read input file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create list of all entries (original + wiki)
    all_entries = []
    
    # Add original entries
    all_entries.extend(data)
    num_mcq = len(all_entries)
    
    # Calculate max number of wiki entries to maintain 20% ratio
    # If x is wiki entries and num_mcq is MCQ entries:
    # x / (x + num_mcq) <= 0.2
    # Solving for x: x <= 0.25 * num_mcq
    max_wiki_entries = int(num_mcq * 0.25)  # This gives us 20% of total entries
    
    # Add WikiText entries (up to max_wiki_entries)
    num_wiki = 0
    for text in wiki_texts:
        if num_wiki >= max_wiki_entries:
            break
            
        # Skip empty or whitespace-only texts
        if not text or text.isspace():
            continue
            
        # Create new entry with WikiText
        wiki_entry = {
            "prompt": text,
            "prompt_metadata": {}
        }
        all_entries.append(wiki_entry)
        num_wiki += 1
    
    # Print statistics
    total = num_mcq + num_wiki
    print(f"\nFile: {os.path.basename(input_path)}")
    print(f"MCQ entries: {num_mcq} ({num_mcq/total*100:.1f}%)")
    print(f"WikiText entries: {num_wiki} ({num_wiki/total*100:.1f}%)")
    print(f"Total entries: {total}")
    
    # Shuffle all entries
    random.shuffle(all_entries)
    
    # Write to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_entries, f, indent=2)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load WikiText dataset
    print("Loading WikiText dataset...")
    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")
    wiki_texts = wikitext["text"]
    print(f"Loaded {len(wiki_texts)} WikiText entries")
    
    # Create output directories and process files
    for dataset in ['csqa', 'mmlu']:
        input_dir = f"{dataset}_quoted_qa"
        output_dir = f"{dataset}_quoted_qa_wiki"
        
        # Get all json files in input directory
        input_files = glob.glob(f"{input_dir}/*.json")
        
        for input_file in input_files:
            # Generate output path
            filename = Path(input_file).name
            output_file = os.path.join(output_dir, filename)
            
            # Process file
            process_file(input_file, output_file, wiki_texts)
            print(f"Processed {input_file} -> {output_file}")

if __name__ == "__main__":
    main() 