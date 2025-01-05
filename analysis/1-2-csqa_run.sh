#!/bin/bash
#SBATCH --job-name=csqa_quoted
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=0-20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x.stdout
#SBATCH --error=slurm_logs/%x.stderr

# Runs for a single model

echo "Starting job $SLURM_JOB_ID"

# Check if model name was provided
if [ -z "$1" ]; then
    echo "Error: Model name must be provided as argument"
    echo "Usage: sbatch $0 <model_name>"
    exit 1
fi

model="$1"
echo "Using model: $model"

echo "loading modules"
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

echo "Starting script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

PYTHON_EXECUTABLE=/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/thesis/bin/python

MAX_NEW_TOKENS=50
results_dir="set-based-prompting-finetuning/results/csqa_quoted"
input_dir="set-based-prompting-finetuning/data/csqa_quoted"

model_path_safe=$(echo $model | sed 's/\//_/g')

echo "Running model $model"

$PYTHON_EXECUTABLE -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

nvidia-smi

cd ..

mkdir -p $results_dir
model_results_dir=$results_dir/$model_path_safe-$MAX_NEW_TOKENS
mkdir -p $model_results_dir

# List all JSON files and count them
echo "Available input files in ${input_dir}:"
ls ${input_dir}/*.json
file_count=$(ls ${input_dir}/*.json | wc -l)
echo "Found $file_count files to process"

# Process each file
for input_file in ${input_dir}/*.json; do
    start_time=$(date +%s)
    echo "Running model $model on $input_file"

    basename=$(basename "$input_file")
    fname_safe=${basename%.json}

    $PYTHON_EXECUTABLE set-based-prompting-finetuning/main.py \
        --model-name "$model" \
        --torch-device cuda \
        --max-new-tokens $MAX_NEW_TOKENS \
        --include-probs \
        --temp_file \
        --append-temp-file \
        --infile "$input_file" \
        --outfile "$model_results_dir/$model_path_safe-$MAX_NEW_TOKENS-$fname_safe.jsonl"

    end_time=$(date +%s)
    echo "Model $model took $((end_time - start_time)) seconds to run on $input_file"
done 