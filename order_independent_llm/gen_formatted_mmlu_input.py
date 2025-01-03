import json
import os
import argparse
import pandas as pd

answer_index_map={"A":1,"B":2,"C":3,"D":4,"E":5}

def gen_formatted_input_file(infile,outfile):
    df=pd.read_csv(infile,header=None)
    outputs=[]

    for i in range(df.shape[0]):
        row=df.iloc[i]
        suffix=" Answer: "
        prompt=row[0]+"<|start_2d|>"+"<|split_2d|>".join(row[1:-1])+f"<|end_2d|>{suffix}"
        answer=row.tolist()[-1]
        answer_index=answer_index_map[answer]
        label=row[answer_index]
        incorrect_answers=row[1:answer_index].tolist()+row[answer_index+1:-1].tolist()
        outputs.append({
            "prompt":prompt,
            "prompt_metadata":{
                "label":label,
                "incorrect_answers":incorrect_answers,
            },
        })
    with open(outfile, "w") as f:
        json.dump(outputs, f)

def gen_formatted_mmlu_files(mmlu_src_dir,mmlu_tgt_dir):
    for fname in os.listdir(mmlu_src_dir):
        out_fname = fname.replace(".csv",".json")
        gen_formatted_input_file(f"{mmlu_src_dir}/{fname}",f"{mmlu_tgt_dir}/{out_fname}")

# python triple_queries_order_dependence/order_independent_llm/gen_formatted_mmlu_input.py --mmlu-src-dir data/test --mmlu-tgt-dir triple_queries_order_dependence/data/mmlu
#
def main():
    parser = argparse.ArgumentParser(
        description="Run attention mask editing tests on a given model"
    )
    parser.add_argument(
        "--mmlu-src-dir",
        type=str,
        help="path to directory of mmlu test data files",
    )
    parser.add_argument(
        "--mmlu-tgt-dir",
        type=str,
        help="path to directory of mmlu test data files",
    )

    args = parser.parse_args()
    mmlu_src_dir: str = args.mmlu_src_dir
    mmlu_tgt_dir: str = args.mmlu_tgt_dir
    gen_formatted_mmlu_files(mmlu_src_dir,mmlu_tgt_dir)

if __name__ == "__main__":
    main()
