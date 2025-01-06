#!/bin/bash
#SBATCH --job-name=sbp_finetune
##SBATCH --partition=seas_gpu
#SBATCH --partition=gpu_requeue
###SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
###SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1 # See of run can get allocated with only 1 GPU
##SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --gres=gpu:4
#SBATCH --constraint='(a40|v100|a100|h100)'
#SBATCH --time=0-12:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Exit on any error
set -e

# Function to cleanup temporary files
cleanup() {
    local exit_code=$?
    echo "Cleaning up..."
    if [ -f "${FSDP_CONFIG_FILE}" ]; then
        rm "${FSDP_CONFIG_FILE}"
    fi
    echo "Script ended at: $(date) with exit code ${exit_code}"
    exit $exit_code
}

# Sample: sbatch finetune.sh False mmlu_quoted
# Sample: sbatch finetune.sh False mmlu_quoted_qa
# Sample: sbatch finetune.sh False mmlu_quoted_qa_wiki
# Sample: sbatch finetune.sh False mmlu_quoted_qa_s2d csqa_quoted_s2d mmlu_quoted_s2d

# Register cleanup function to run on script exit
trap cleanup EXIT

# Log start time
echo "Script started at: $(date)"

# Parse command line arguments
MODEL_NAME=${5:-"meta-llama/Llama-2-7b-hf"}  # Default to Llama-2-7b-hf if not provided
IS_LORA=${1:-"False"}  # New parameter, defaults to False
TRAIN_DATA=${2:-"mmlu_quoted.jsonl"}  # New parameter for training data
# add command line argument for test_dir
TEST_DIR_CSQA=${3:-"csqa_quoted"}  # New parameter for test directory
TEST_DIR_MMLU=${4:-"mmlu_quoted"}  # New parameter for test directory
# Generate run datetime
RUN_DATETIME=$(date +%Y%m%d-%H%M%S)

# Validate IS_LORA
if [ "${IS_LORA}" != "True" ] && [ "${IS_LORA}" != "False" ]; then
    echo "Error: IS_LORA must be either 'True' or 'False'"
    exit 1
fi

# Add timeout value as a variable at the top
NVIDIA_SMI_TIMEOUT=10

sleep 20

# echo output of nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Improve the GPU check function
check_gpus() {
    for i in {1..5}; do  # Increased from 3 to 5 attempts
        if gpu_count=$(timeout ${NVIDIA_SMI_TIMEOUT} nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l); then
            echo "Found ${gpu_count} GPUs"
            #if [ $gpu_count -lt 4 ]; then
            #    echo "Error: Script requires 4 GPUs, but only ${gpu_count} found"
            #    exit 1
            # fi
            return 0
        fi
        echo "Attempt $i: Waiting for nvidia-smi... (${NVIDIA_SMI_TIMEOUT}s timeout)"
        sleep 5  # Increased from 2s to 5s
    done
    echo "Error: Failed to get GPU count after 5 attempts"
    exit 1
}

# Use the function
check_gpus

# Create slurm_logs directory if it doesn't exist
if ! mkdir -p slurm_logs; then
    echo "Error: Failed to create slurm_logs directory"
    exit 1
fi

# Define base directory
BASE_DIR="/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp" # /n/holylabs/LABS/dwork_lab/Lab/katrinabrown/

# create the BASE_DIR if it doesn't exist
if ! mkdir -p "${BASE_DIR}"; then
    echo "Error: Failed to create base directory ${BASE_DIR}"
    exit 1
fi

# Get current datetime in YYYYMMDD-HHmmSS format
MODEL_NAME="meta-llama/Llama-2-7b-hf"
model_path_safe=$(echo $MODEL_NAME | sed 's/\//_/g')
# Include IS_LORA in output directory name
OUTPUT_DIR="${BASE_DIR}/${MODEL_NAME}/${train_data_safe}/${RUN_DATETIME}-${IS_LORA}"

# Create run-specific benchmarks file
BENCHMARKS_FILE="${OUTPUT_DIR}/benchmarks.jsonl"

# Validate input files exist
if [ ! -f "data/mmlu_quoted.jsonl" ]; then
    echo "Error: Training dataset file not found"
    exit 1
fi

# Create output directory with error checking
if ! mkdir -p "${OUTPUT_DIR}"; then
    echo "Error: Failed to create output directory ${OUTPUT_DIR}"
    exit 1
fi

if ! [ -w "${OUTPUT_DIR}" ]; then
    echo "Error: No write permission for output directory ${OUTPUT_DIR}"
    exit 1
fi

# Echo variable values
echo "MODEL_NAME: ${MODEL_NAME}"
echo "DATETIME: ${RUN_DATETIME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "IS_LORA: ${IS_LORA}"
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "TEST_DIR_CSQA: ${TEST_DIR_CSQA}"
echo "TEST_DIR_MMLU: ${TEST_DIR_MMLU}"
echo "Current directory: $(pwd)"

# Get GPU count using nvidia-smi
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Found ${GPU_COUNT} GPUs"

# Set environment variables for FSDP
export ACCELERATE_USE_FSDP="true"
export FSDP_TRANSFORMER_CLS_TO_WRAP="LlamaDecoderLayer"
export FSDP_BACKWARD_PREFETCH="BACKWARD_PRE"
export FSDP_FORWARD_PREFETCH="true"
export FSDP_SYNC_MODULE_STATES="true"
export FSDP_USE_ORIG_PARAMS="true"
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
export FSDP_OFFLOAD_PARAMS="false"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM="false"

# Select training script based on IS_LORA
TRAIN_SCRIPT="scripts/train.py"
if [ "${IS_LORA}" = "True" ]; then
    TRAIN_SCRIPT="scripts/train.py"
    echo "Using LoRA training script: ${TRAIN_SCRIPT}"
else
    TRAIN_SCRIPT="scripts/train_full.py"
    echo "Using full model training script: ${TRAIN_SCRIPT}"
fi

# Run the training script with proper GPU count
if ! torchrun \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    ${TRAIN_SCRIPT} \
    --model_name_or_path "${MODEL_NAME}" \
    --model_checkpoint "${MODEL_NAME}" \
    --data_path "data/${TRAIN_DATA}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --gradient_checkpointing True \
    --run_name "${model_path_safe}-finetuning-${RUN_DATETIME}" \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --cache_dir "${BASE_DIR}/cache/transformers"
then
    echo "Error: Training failed"
    exit 1
fi

# After training completes successfully
echo "Training completed, creating latest symlink..."

# Create model directory and parent directories if they don't exist
model_dir="${BASE_DIR}/${model_path_safe}"
mkdir -p "${model_dir}"

# Create symlink to latest run
rm -f "${model_dir}/latest"
ln -s "${RUN_DATETIME}" "${model_dir}/latest"

# echo the path to the generate symlink
echo "Symlink path: ${model_dir}/latest"

# Define the final weights directory path
FINAL_WEIGHTS_DIR="${OUTPUT_DIR}/final_weights"

echo "Submitting CSQA evaluation job..."

# Submit CSQA evaluation job with final weights directory
csqa_job_id=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
    --parsable \
    --output=slurm_logs/csqa_eval_%j.out \
    --error=slurm_logs/csqa_eval_%j.err \
    --export=ALL \
    --wrap="source /n/home11/katrinabrown/.bashrc && \
           conda activate thesis && \
           module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01 && \
           bash analysis/1-2-csqa_run.sh \"${FINAL_WEIGHTS_DIR}\" \"${TEST_DIR_CSQA}\"")

if [ $? -ne 0 ] || ! [[ "$csqa_job_id" =~ ^[0-9]+$ ]]; then
    echo "Error: Failed to submit CSQA evaluation job"
    exit 1
fi

echo "Submitting MMLU evaluation job..."

# Submit MMLU evaluation job with final weights directory
mmlu_job_id=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
    --parsable \
    --output=slurm_logs/mmlu_eval_%j.out \
    --error=slurm_logs/mmlu_eval_%j.err \
    --export=ALL \
    --wrap="source /n/home11/katrinabrown/.bashrc && \
           conda activate thesis && \
           module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01 && \
           bash analysis/2-2-mmlu_run.sh \"${FINAL_WEIGHTS_DIR}\" \"${TEST_DIR_MMLU}\"")

if [ $? -ne 0 ] || ! [[ "$mmlu_job_id" =~ ^[0-9]+$ ]]; then
    echo "Error: Failed to submit MMLU evaluation job"
    exit 1
fi

echo "Submitting benchmark jobs..."

# Submit benchmarking jobs through benchmark_models.sh
benchmark_job_id=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
    --parsable \
    --output=slurm_logs/benchmark_chain_%j.out \
    --error=slurm_logs/benchmark_chain_%j.err \
    --export=ALL \
    --wrap="source /n/home11/katrinabrown/.bashrc && \
           conda activate benchmark_env && \
           module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01 && \
           export PATH=\"/n/holylabs/LABS/dwork_lab/Lab/katrinabrown/home/conda/envs/benchmark_env/bin:$PATH\" && \
           bash scripts/benchmark_models.sh \"${OUTPUT_DIR}\" && \
           conda activate thesis")

if [ $? -ne 0 ] || ! [[ "$benchmark_job_id" =~ ^[0-9]+$ ]]; then
    echo "Error: Failed to submit benchmark jobs"
    exit 1
fi

echo "Submitted CSQA evaluation job with ID: ${csqa_job_id}"
echo "Submitted MMLU evaluation job with ID: ${mmlu_job_id}"
echo "Submitted benchmark job chain with ID: ${benchmark_job_id}"
echo "Output directory: ${OUTPUT_DIR}"
echo "All jobs submitted. Job chain:"
echo "Training job (${SLURM_JOB_ID}) -> CSQA eval (${csqa_job_id}), MMLU eval (${mmlu_job_id}), Benchmark jobs (${benchmark_job_id})"

