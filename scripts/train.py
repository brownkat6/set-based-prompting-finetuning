import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any, Tuple, Union, Callable
import os
from datetime import datetime, timedelta
from contextlib import nullcontext
import numpy as np

import torch
import transformers
import utils
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, LlamaForCausalLM, HfArgumentParser, AutoTokenizer
import torch.backends.mps
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from torch.cuda.amp import autocast
from accelerate.optimizer import AcceleratedOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import order_independent_llm

# Finetune with LORA

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "}"
DEFAULT_BOS_TOKEN = "{"
DEFAULT_UNK_TOKEN = "<unk>"

from train_full import get_parallel_inputs, make_supervised_data_module, DataArguments, TrainingArguments
from train_full import smart_tokenizer_and_embedding_resize
from train_full import preprocess
from train_full import SupervisedDataset
from train_full import DataCollatorForSupervisedDataset
from train_full import SubsetDatasetWithAttrs
from train_full import TrustedTrainer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_checkpoint: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "Model checkpoint to use for PCA components"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    is_order_independent: bool = field(default=True, metadata={"help": "Whether to use order-independent processing"})

def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Disable SDPA by setting environment variable
    os.environ["USE_TORCH_SDPA"] = "0"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    
    # GPU and dtype setup
    try:
        torch.cuda.init()
        torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16
            n_gpus = torch.cuda.device_count()
            print(f"Found {n_gpus} CUDA devices")
            for i in range(n_gpus):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        else:
            print("Warning: No GPU found. Training will be slow!")
            device = torch.device("cpu")
            dtype = torch.float32
    except Exception as e:
        print(f"Error during CUDA initialization: {e}")
        print("Falling back to CPU due to CUDA error")
        device = torch.device("cpu")
        dtype = torch.float32
    
    # Modify training arguments based on device
    if device.type == "cpu" or (device.type == "cuda" and torch.cuda.device_count() <= 1):
        print(f"Disabling FSDP for CPU/single-GPU training")
        os.environ["ACCELERATE_USE_FSDP"] = "false"
        training_args.fsdp = ""
        training_args.fsdp_config = None

    try:
        # Load base model with SDPA disabled
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
        )
        model = model.to(device)
        
        # Verify SDPA is disabled
        if hasattr(model.config, 'use_sdpa'):
            try:
                print(f"Model config has use_sdpa: {model.config.use_sdpa}")
                model.config.use_sdpa = False
            except Exception as e:
                print(f"Model config has no use_sdpa attribute: {e}")
        if hasattr(model.config, 'use_flash_attention'):
            try:
                print(f"Model config has use_flash_attention: {model.config.use_flash_attention}")
                model.config.use_flash_attention = False
            except Exception as e:
                print(f"Model config has no use_flash_attention attribute: {e}")
        
        #print(f"Model config has use_sdpa: {model.config.use_sdpa}")
        try:
            print(f"model.model._use_sdpa: {model.model._use_sdpa}")
            model.model._use_sdpa = False # set sdpa to false
            print(f"model.model._use_sdpa: {model.model._use_sdpa}")
        except Exception as e:
            print(f"model.model has no _use_sdpa attribute: {e}")
            
        print("SDPA has been disabled")
        
        print("\n=== Initializing LoRA ===")
        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Verify model parameters before LoRA
        print("\nPre-LoRA Parameter Check:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        
        print("\nPreparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        
        #print(f"Overriding model for order independent training")
        model = order_independent_llm.input_processing.get_2D_attention_accepting_model(model)
        #print(model._update_model_kwargs_for_generation)
        
        print("\nApplying LoRA...")
        model = get_peft_model(model, config)
        
        print(f"Overriding peft model for order independent training")
        #print(model.base_model.model._update_model_kwargs_for_generation)
        model = order_independent_llm.input_processing.get_2D_attention_accepting_model(model)
        #print(model.base_model.model._update_model_kwargs_for_generation)
                
        # Print trainable parameters
        model.print_trainable_parameters()

    except Exception as e:
        print(f"Failed to load/prepare model: {e}")
        raise

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Save initial weights
    try:
        initial_weights_dir = os.path.join(training_args.output_dir, "initial_weights")
        os.makedirs(initial_weights_dir, exist_ok=True)
        if not any(os.scandir(initial_weights_dir)):
            # For LoRA, we need to save both base model and adapter weights
            model.save_pretrained(initial_weights_dir)  # This saves LoRA weights
            tokenizer.save_pretrained(initial_weights_dir)
            # Also save the base model config
            model.base_model.config.save_pretrained(initial_weights_dir)
            logging.info(f"Saved initial model weights to {initial_weights_dir}")
    except Exception as e:
        logging.error(f"Failed to save initial weights: {e}")
        raise

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model=model,
    )
    
    trainer = TrustedTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )
    
    print("=== DEBUG: Checking sample batch ===")
    first_batch = next(iter(data_module["train_dataset"]))
    # Inspect shape and example token indices
    print("input_ids:", first_batch["input_ids"])
    print("labels:", first_batch["labels"])
    # Then run a quick forward pass manually
    with torch.set_grad_enabled(True):
        outputs = model(
            input_ids=first_batch["input_ids"].unsqueeze(0).to(device),
            labels=first_batch["labels"].unsqueeze(0).to(device)
        )
        print(outputs.keys())
        print(outputs.loss, outputs.loss.requires_grad, outputs.logits.shape)

    # Verify trainer setup
    print("\nVerifying trainer configuration:")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Training with LoRA: {isinstance(model, PeftModel)}")
    print(f"Number of training examples: {len(data_module['train_dataset'])}")
    if data_module['eval_dataset']:
        print(f"Number of validation examples: {len(data_module['eval_dataset'])}")

    # Add timing callback
    from train_full import TimingCallback

    # Add callback to trainer
    trainer.add_callback(TimingCallback(print_interval_steps=100))
    
    print("Training model...")
    trainer.create_optimizer()
    print("Optimizer defaults: ", trainer.optimizer.defaults)
    trainer.train()
    print("Saving model state...")
    trainer.save_state()
    
    # Save final weights - ensure proper LoRA adapter saving
    print("Saving final model weights...")
    final_weights_dir = os.path.join(training_args.output_dir, "final_weights")
    os.makedirs(final_weights_dir, exist_ok=True)
    
    try:
        # Save LoRA adapter weights and config
        model.save_pretrained(final_weights_dir)
        # Save tokenizer and base model config
        tokenizer.save_pretrained(final_weights_dir)
        model.base_model.config.save_pretrained(final_weights_dir)
        logging.info(f"Saved final LoRA weights to {final_weights_dir}")
    except Exception as e:
        logging.error(f"Failed to save final weights: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train()
