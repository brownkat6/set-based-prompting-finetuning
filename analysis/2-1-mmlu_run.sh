#!/bin/bash
#SBATCH --job-name=mmlu_quoted
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=0-20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

##SBATCH --open-mode=truncate
##SBATCH --output=slurm_logs/%j-%a-%x.stdout
##SBATCH --error=slurm_logs/%j-%a-%x.stderr

#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x-%a.stdout
#SBATCH --error=slurm_logs/%x-%a.stderr

#SBATCH --array=0-6
#set -e

echo "Starting job $SLURM_JOB_ID"

echo "loading modules"
# Load modules
# conda activate cuda_11
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

echo "Starting script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"

# Set this to the path of your python executable
PYTHON_EXECUTABLE=/n/holylabs/LABS/dwork_lab/Lab/reidmcy/home/conda/envs/cuda_11/bin/python

MAX_NEW_TOKENS=50
results_dir="results/mmlu_quoted"
input_dir="data/mmlu_quoted"

# NOTE: can modify the models array to contain a list of paths to model weights (e.g. weights for a finetuned gpt model)
models=(
    "gpt2"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    #"WizardLM/WizardLM-7B-V1.0"
    #"lmsys/vicuna-7b-v1.5"
    "meta-llama/Meta-Llama-3-8B"
    "mistralai/Mistral-7B-Instruct-v0.2"
)

model=${models[$SLURM_ARRAY_TASK_ID]}

#strip / from model name
model_path_safe=$(echo $model | sed 's/\//_/g')

echo "Running model $model"

/n/holylabs/LABS/dwork_lab/Lab/reidmcy/home/conda/envs/cuda_11/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

nvidia-smi

cd ..

mkdir -p $results_dir

model_results_dir=$results_dir/$model_path_safe-$MAX_NEW_TOKENS

mkdir -p $model_results_dir

for fname in $input_dir/*.json; do
    start_time=$(date +%s)
    echo "Running model $model on $fname"

    basename=$(basename $fname)
    #strip .json from filename
    fname_safe=${basename%.json}

    $PYTHON_EXECUTABLE main.py \
        --model-name $model \
        --torch-device cuda \
        --max-new-tokens $MAX_NEW_TOKENS \
        #--include-probs \
        --temp_file \
        --append-temp-file \
        --infile $fname \
        --outfile $model_results_dir/$model_path_safe-$MAX_NEW_TOKENS-$fname_safe.jsonl

    end_time=$(date +%s)
    echo "Model $model took $((end_time - start_time)) seconds to run on $fname"
done