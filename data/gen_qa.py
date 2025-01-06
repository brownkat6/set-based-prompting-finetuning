import json
import os
import glob
from pathlib import Path

def process_file(input_path: str, output_path: str):
    """Process a single input file and write modified version to output path"""
    
    # Read input file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Modify each entry to append the answer
    for entry in data:
        entry["prompt"] = entry["prompt"] + entry["prompt_metadata"]["label"]
    
    # Write to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    # Create output directories
    for dataset in ['csqa', 'mmlu']:
        input_dir = f"{dataset}_quoted"
        output_dir = f"{dataset}_quoted_qa"
        
        # Get all json files in input directory
        input_files = glob.glob(f"{input_dir}/*.json")
        
        for input_file in input_files:
            # Generate output path
            filename = Path(input_file).name
            output_file = os.path.join(output_dir, filename)
            
            # Process file
            process_file(input_file, output_file)
            print(f"Processed {input_file} -> {output_file}")

if __name__ == "__main__":
    main() 