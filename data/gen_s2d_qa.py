import json
import os
import glob
from pathlib import Path

def process_file(input_path: str, output_path_qa: str, output_path_noqa: str):
    """Process a single input file and write modified versions to output paths"""
    
    # Read input file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create copies for QA and non-QA versions
    data_qa = data.copy()
    data_noqa = data.copy()
    
    # Modify each entry to add s2d/e2d markers
    for entry in data_qa:
        # Add s2d before start_2d and e2d after end_2d
        entry["prompt"] = entry["prompt"].replace("<|start_2d|>", "*tag*<|start_2d|>")
        entry["prompt"] = entry["prompt"].replace("<|end_2d|>", "<|end_2d|>*etag*")
    
    # Modify non-QA entries (remove answer and add s2d/e2d markers)
    for entry in data_noqa:
        # Remove the answer from the prompt
        if "Answer: " in entry["prompt"]:
            entry["prompt"] = entry["prompt"].split("Answer: ")[0] + "Answer: "
        # Add s2d before start_2d and e2d after end_2d
        #entry["prompt"] = entry["prompt"].replace("<|start_2d|>", "|s2d|<|start_2d|>")
        #entry["prompt"] = entry["prompt"].replace("<|end_2d|>", "<|end_2d|>|e2d|")
    
    # Write to output files
    os.makedirs(os.path.dirname(output_path_qa), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_noqa), exist_ok=True)
    
    with open(output_path_qa, 'w') as f:
        json.dump(data_qa, f, indent=2)
        
    with open(output_path_noqa, 'w') as f:
        json.dump(data_noqa, f, indent=2)

def main():
    # Create output directories
    for dataset in ['csqa', 'mmlu']:
        input_dir = f"{dataset}_quoted_qa"
        output_dir_qa = f"{dataset}_quoted_qa_s2d"
        output_dir_noqa = f"{dataset}_quoted_s2d"
        
        # Get all json files in input directory
        input_files = glob.glob(f"{input_dir}/*.json")
        
        for input_file in input_files:
            # Generate output paths
            filename = Path(input_file).name
            output_file_qa = os.path.join(output_dir_qa, filename)
            output_file_noqa = os.path.join(output_dir_noqa, filename)
            
            # Process file
            process_file(input_file, output_file_qa, output_file_noqa)
            print(f"Processed {input_file} -> {output_file_qa} and {output_file_noqa}")

if __name__ == "__main__":
    main() 