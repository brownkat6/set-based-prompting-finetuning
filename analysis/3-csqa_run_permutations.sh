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

#SBATCH --array=0-0

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
results_dir="set-based-prompting-finetuning/results/csqa_quoted_permutations"
input_dir="set-based-prompting-finetuning/data/csqa_quoted"

models=(
    "meta-llama/Llama-2-7b-hf"
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

for fname in $input_dir/*.json; do
    echo "Running $fname"
    start_time=$(date +%s)
    basename=$(basename $fname)
    #strip .json from filename
    fname_safe=${basename%.json}
    $PYTHON_EXECUTABLE set-based-prompting-finetuning/main.py \
        --model-name $model \
        --torch-device cuda \
        --max-new-tokens $MAX_NEW_TOKENS \
        --num-normal-ordering-permutations 4 \
        --include-probs \
        --temp_file \
        --infile $fname \
        --outfile $model_results_dir/$model_path_safe-permutations-$MAX_NEW_TOKENS-$fname_safe.jsonl
    end_time=$(date +%s)
    echo "Model $model took $((end_time - start_time)) seconds to run on $fname_safe"
done 