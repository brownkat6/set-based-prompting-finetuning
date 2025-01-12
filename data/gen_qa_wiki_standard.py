import json
import os
from pathlib import Path

def process_file(input_path, output_path):
       # Read input file as a JSON array
   with open(input_path, 'r', encoding='utf-8') as f:
       try:
           data_list = json.load(f)
           
           # Process each entry in the JSON array
           for entry in data_list:
               if isinstance(entry, dict) and 'prompt' in entry:
                   # Get the prompt
                   prompt = entry['prompt']
                   
                   # Remove the special tags
                   prompt = prompt.replace('<|split_2d|>', '')
                   prompt = prompt.replace('<|start_2d|>', '')
                   prompt = prompt.replace('<|end_2d|>', '')
                   
                   # Add back the enclosing tags
                   prompt = '<|start_2d|>' + prompt + '<|end_2d|>'
                   
                   # Update the prompt in the entry
                   entry['prompt'] = prompt
           
           # Create output directory if it doesn't exist
           os.makedirs(os.path.dirname(output_path), exist_ok=True)
           
           # Write the processed JSON array to output file
           with open(output_path, 'w', encoding='utf-8') as outf:
               json.dump(data_list, outf, ensure_ascii=False, indent=2)
               
       except json.JSONDecodeError as e:
           print(f"Error processing {input_path}: {e}")

def main():
    # Process both mmlu and csqa files
    # base_dir = Path('data')
    
    for dataset in ['mmlu_quoted_qa_wiki', 'csqa_quoted_qa_wiki']:
        input_dir = Path(dataset)
        output_dir = Path(f'{dataset}_standard')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all json files in the input directory
        if input_dir.exists():
            for file_path in input_dir.glob('*.json'):
                output_path = output_dir / file_path.name
                process_file(file_path, output_path)
                print(f"Processed {file_path} -> {output_path}")
        else:
            print(f"Input directory {input_dir} does not exist")

if __name__ == '__main__':
    main() 