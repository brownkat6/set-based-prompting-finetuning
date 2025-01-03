#!/bin/bash
#SBATCH --job-name=csqa_70B_split_array
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --time=0-20:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%j-%a-%x.stdout
#SBATCH --error=slurm_logs/%j-%a-%x.stderr

##SBATCH --open-mode=append
#SBATCH --open-mode=truncate
#SBATCH --array=0-3
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

MAX_NEW_TOKENS=100
results_dir="results/csqa_split"

/n/holylabs/LABS/dwork_lab/Lab/reidmcy/home/conda/envs/cuda_11/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

nvidia-smi

cd ..

mkdir -p $results_dir



models=(
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/llama-2-70b-chat-hf"
    "meta-llama/Meta-Llama-3-70B"
    "meta-llama/Meta-Llama-3-70B-Instruct"
)

model=${models[$SLURM_ARRAY_TASK_ID]}

#strip / from model name
model_path_safe=$(echo $model | sed 's/\//_/g')

model_results_dir=$results_dir/$model_path_safe-$MAX_NEW_TOKENS
mkdir -p $model_results_dir


for fname in data/csqa_split/*; do
    start_time=$(date +%s)
    echo "Running model $model on $fname"

    basename=$(basename $fname)
    #strip .json from filename
    fname_safe=${basename%.json}

    $PYTHON_EXECUTABLE main.py \
        --model-name $model \
        --torch-device auto \
        --max-new-tokens $MAX_NEW_TOKENS \
        --add-only-parallel-attention \
        --add-only-parallel-position \
        --temp_file \
        --append-temp-file \
        --infile $fname \
        --outfile $model_results_dir/$model_path_safe-$MAX_NEW_TOKENS-$fname_safe.jsonl

    end_time=$(date +%s)
    echo "Model $model took $((end_time - start_time)) seconds to run on $fname"
done
