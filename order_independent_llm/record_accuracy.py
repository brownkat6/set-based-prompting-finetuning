import order_independent_llm
import os
import argparse



def main():
    parser = argparse.ArgumentParser(
        description="Run attention mask editing tests on a given model"
    )
    parser.add_argument(
        "--model-output-file",
        type=str,
        help="path/to/file containing model outputs",
        required=False,
        default="",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        help="path/to/dir containing model outputs",
        required=False,
        default="",
    )
    parser.add_argument(  
        "--accuracy-file",
        type=str,
        help="path/to/csv to record experiment accuracy results",
        required=False,
        default="records.csv",
    )
    parser.add_argument(
        "--scoring-mode",
        type=str,
        required=True,
        choices=["max_token_prob","pct_raw_output_contains_correct_answer_only"],
        help="How to determine whether a model output is \'correct\' or not. If max_token_prob, then compute the probability that the model outputs the correct answer vs the prob that the model outputs any of the incorrect answers. If pct_raw_output_contains_correct_answer_only, then determine whether the raw model output contains the correct answer text and none of the incorrect answer texts.",
    )
    args = parser.parse_args()
    model_output_file: str = args.model_output_file
    model_output_dir: str = args.model_output_dir
    accuracy_file: str = args.accuracy_file

    model_output_files=[]
    if model_output_file:
        model_output_files.append(model_output_file)
    if model_output_dir:
        model_output_files+=list([f"{model_output_dir}/{fo}" for fo in os.listdir(model_output_dir) if ".json" in fo])
    for output_file_name in model_output_files:
        order_independent_llm.print_with_timestamp(
            f"Recording accuracy of model outputs for {output_file_name}, accuracy records saved to {accuracy_file}"
        )
        order_independent_llm.record_accuracy(
            output_file_name, accuracy_file
        )
    
# $ python3 record_accuracy.py --model-output-file data/csqa_output.json
# $ python3 record_accuracy.py --model-output-dir data/csqa_output.json --accuracy-file data/mmlu_output/records.csv
# python3 record_accuracy.py --model-output-dir data/mmlu_output --accuracy_file data/mmlu_output/records.csv
if __name__ == "__main__":
    main()



