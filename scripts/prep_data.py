import json
import glob
import os
from pathlib import Path

def combine_json_files(input_dir: str, output_file: str) -> None:
    """
    Combines all JSON files in input_dir into a single JSONL file.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Path to output JSONL file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
        
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Combine all files into one JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for json_file in sorted(json_files):
            print(f"Processing {json_file}")
            with open(json_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                # Write each entry as a separate line
                if isinstance(data, list):
                    for item in data:
                        json.dump(item, outfile, ensure_ascii=False)
                        outfile.write('\n')
                else:
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
    
    print(f"Combined data written to {output_file}")

def main():
    # Get the path to the root directory (one level up from scripts/)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    
    # Get all subdirectories in the data directory
    subdirs = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d))]
    
    # Process each subdirectory
    for subdir in subdirs:
        input_dir = os.path.join(data_dir, subdir)
        output_file = os.path.join(data_dir, f"{subdir}.jsonl")
        
        print(f"\nProcessing {subdir}...")
        combine_json_files(input_dir, output_file)

if __name__ == "__main__":
    main()
