import json
import os
import argparse
import pandas as pd
from promptsource.templates import DatasetTemplates
from datasets import load_dataset

answer_index_map={"A":1,"B":2,"C":3,"D":4,"E":5}

def load_csqa_prompts():
    # Load dataset and format into prompts
    commonsense_qa_prompts = DatasetTemplates('commonsense_qa')
    prompt=commonsense_qa_prompts["question_answering"]
    dataset = load_dataset("commonsense_qa", split="train")

    # Apply prompt to all examples in dataset
    prompts,labels=[],[]
    for example in dataset:
        p,label=prompt.apply(example)
        prompts.append(p)
        labels.append(label)
    assert(len(prompts)==len(labels))
    return prompts,labels

def gen_formatted_csqa_input_file(outfile):
    prompts,labels=load_csqa_prompts()
    incorrect_answers=[]
    for p,label in zip(prompts,labels):
        options=p.split('\n- ')[1:]
        incorrect_answers.append([o for o in options if o!=label])
    assert(len(incorrect_answers)==len(prompts))
    
    # Format prompts into order_independent format of f"{prefix}<|start_2d|>parallel_substrings<|end_2d|>suffix" for each prompt
    formatted_inputs=[]
    for i,p in enumerate(prompts):
        prefix=p.split("\n")[0]+"\n"
        parallel=p.split("\n- ")[1:]
        parallel=["\n- "+s for s in parallel]
        formatted_inputs.append({
            "prompt":prefix+"<|start_2d|>"+"<|split_2d|>".join(parallel)+"<|end_2d|> Answer: ",
            "prompt_metadata": {
                "label":labels[i],
                "incorrect_answers":incorrect_answers[i]},
        })
    assert(len(formatted_inputs)==len(prompts))
    with open("data/csqa_input.json", "w") as outfile: 
        json.dump(formatted_inputs, outfile)

# python order_independent_llm/gen_formatted_csqa_input.py --csqa-outfile data/csqa_input.json
def main():
    parser = argparse.ArgumentParser(
        description="Generate a json with CSQA prompts in the process-in-parallel input format"
    )
    parser.add_argument(
        "--csqa-outfile",
        type=str,
        help="path/to/csqa_input.json where the formatted inputs will be saved",
    )
    
    args = parser.parse_args()
    csqa_outfile: str = args.csqa_outfile
    gen_formatted_csqa_input_file(csqa_outfile)

if __name__ == "__main__":
    main()