#!/bin/bash
#SBATCH --job-name=benchmark_experiment
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=0-12:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/benchmark_main_%j.out
#SBATCH --error=slurm_logs/benchmark_main_%j.err

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Define constants
PYTHON_PATH="/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/benchmark_env/bin/python"

# Get the output directory from command line arguments
OUTPUT_DIR=$1

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Please provide output directory as argument"
    echo "Usage: sbatch benchmark_models.sh /path/to/model/directory"
    exit 1
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: Output directory ${OUTPUT_DIR} does not exist"
    exit 1
fi

# Use the same benchmarks file location as finetune.sh
BENCHMARKS_FILE="${OUTPUT_DIR}/benchmarks.jsonl"

# Array to store benchmark job IDs
benchmark_job_ids=()

# Submit initial weights benchmarking if directory exists
if [ -d "${OUTPUT_DIR}/initial_weights" ]; then
    echo "Submitting initial weights benchmarking job..."
    sbatch_cmd="sbatch \
        --job-name=benchmark_initial \
        --partition=seas_gpu \
        --gres=gpu:1 \
        --time=2:00:00 \
        --mem=32G \
        --output=slurm_logs/benchmark_initial_%j.out \
        --error=slurm_logs/benchmark_initial_%j.err \
        --wrap=\"${PYTHON_PATH} scripts/benchmark_model.py \
            --model_path \\\"${OUTPUT_DIR}/initial_weights\\\" \
            --device \\\"cuda\\\"\""
    echo "Command: $sbatch_cmd"
    job_id=$(eval $sbatch_cmd --parsable)
    benchmark_job_ids+=($job_id)
fi

# Submit checkpoint benchmarking jobs
for checkpoint_dir in "${OUTPUT_DIR}"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        checkpoint_num=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
        echo "Submitting checkpoint-${checkpoint_num} benchmarking job..."
        sbatch_cmd="sbatch \
            --job-name=benchmark_checkpoint_${checkpoint_num} \
            --partition=seas_gpu \
            --gres=gpu:1 \
            --time=2:00:00 \
            --mem=32G \
            --output=slurm_logs/benchmark_checkpoint_${checkpoint_num}_%j.out \
            --error=slurm_logs/benchmark_checkpoint_${checkpoint_num}_%j.err \
            --wrap=\"${PYTHON_PATH} scripts/benchmark_model.py \
                --model_path \\\"${checkpoint_dir}\\\" \
                --device \\\"cuda\\\"\""
        echo "Command: $sbatch_cmd"
        job_id=$(eval $sbatch_cmd --parsable)
        benchmark_job_ids+=($job_id)
    fi
done

# Submit final weights benchmarking if directory exists
if [ -d "${OUTPUT_DIR}/final_weights" ]; then
    echo "Submitting final weights benchmarking job..."
    sbatch_cmd="sbatch \
        --job-name=benchmark_final \
        --partition=seas_gpu \
        --gres=gpu:1 \
        --time=2:00:00 \
        --mem=32G \
        --output=slurm_logs/benchmark_final_%j.out \
        --error=slurm_logs/benchmark_final_%j.err \
        --wrap=\"${PYTHON_PATH} scripts/benchmark_model.py \
            --model_path \\\"${OUTPUT_DIR}/final_weights\\\" \
            --device \\\"cuda\\\"\""
    echo "Command: $sbatch_cmd"
    job_id=$(eval $sbatch_cmd --parsable)
    benchmark_job_ids+=($job_id)
fi

if [ ${#benchmark_job_ids[@]} -eq 0 ]; then
    echo "Error: No model checkpoints found to benchmark"
    exit 1
fi

echo "Submitted benchmark jobs with IDs: ${benchmark_job_ids[*]}"
echo "Creating dependency list: ${dependency_list}"
echo "All jobs submitted. Job chain:"
echo "Benchmark jobs (${benchmark_job_ids[*]})"
