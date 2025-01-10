#!/bin/bash
#SBATCH --job-name=csqa_permutations_quoted
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=0-20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x-%a.stdout
#SBATCH --error=slurm_logs/%x-%a.stderr

#SBATCH --array=0-4

echo "Starting job $SLURM_JOB_ID"

echo "loading modules"
# Load modules
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

echo "Starting script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"

# Set this to the path of your python executable
PYTHON_EXECUTABLE=/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/thesis/bin/python

MAX_NEW_TOKENS=50
results_dir="/n/netscratch/dwork_lab/Lab/katrina/set-based-prompting-finetuning/results/csqa_quoted_permutations"
input_dir="set-based-prompting-finetuning/data/csqa_quoted"

models=(
    #"meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

model=${models[$SLURM_ARRAY_TASK_ID]}

#strip / from model name
model_path_safe=$(echo $model | sed 's/\//_/g')

echo "Running model $model"

$PYTHON_EXECUTABLE -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

nvidia-smi

cd ..

mkdir -p $results_dir
model_results_dir=$results_dir/$model_path_safe-$MAX_NEW_TOKENS
mkdir -p $model_results_dir

# Get first 5 files from input directory
readarray -t input_files < <(ls ${input_dir}/*.json | head -n 5)

# List target files and count them
echo "Target input files:"
printf '%s\n' "${input_files[@]}"
file_count=${#input_files[@]}
echo "Found $file_count files to process"

# Process each specified file
for input_file in "${input_files[@]}"; do
    if [ ! -f "$input_file" ]; then
        echo "Warning: File $input_file does not exist, skipping"
        continue
    fi
    
    echo "Running $input_file"
    start_time=$(date +%s)
    basename=$(basename "$input_file")
    fname_safe=${basename%.json}
    
    $PYTHON_EXECUTABLE set-based-prompting-finetuning/main.py \
        --model-name $model \
        --torch-device cuda \
        --max-new-tokens $MAX_NEW_TOKENS \
        --num-normal-ordering-permutations 4 \
        --temp_file \
        --infile $input_file \
        --outfile $model_results_dir/$model_path_safe-permutations-$MAX_NEW_TOKENS-$fname_safe.jsonl
        
    end_time=$(date +%s)
    echo "Model $model took $((end_time - start_time)) seconds to run on $fname_safe"
done 