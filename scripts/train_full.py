import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
#import logging
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
import order_independent_llm

# Finetune by updating all model weights, not using LORA

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "}"
DEFAULT_BOS_TOKEN = "{"
DEFAULT_UNK_TOKEN = "<unk>"

# NOTE: file modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

from order_independent_llm.input_processing import load_model

# NOISE_FN = gen_tag_bits_vector


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_checkpoint: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "Model checkpoint to use for PCA components"}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    #cache_dir: Optional[str] = field(
    #    default=None,
    #    metadata={"help": "Path to cache directory for storing model and data"}
    #)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=64,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to adopt during training."},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The save strategy to adopt during training."},
    )

    def __post_init__(self):
        # Make sure to call parent's post init
        super().__post_init__()
        # Ensure find_unused_parameters is False for DDP
        self.ddp_find_unused_parameters = False


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, List[Union[torch.Tensor, Dict[str, Any]]]]:
    """Preprocess the data by tokenizing and generating parallel processing tensors."""
    input_ids = []
    labels = []
    attention_masks = []
    position_ids = []
    
    max_length = 64  # Hard-code max length to 64
    
    for source in sources:
        # Get parallel processing tensors
        inputs = get_parallel_inputs(source, tokenizer)
        
        # Truncate all tensors to max_length
        for key in ['input_ids', 'attention_mask', 'position_ids']:
            if key in inputs:
                if inputs[key].size(-1) > max_length:
                    if key == 'attention_mask':
                        # For 4D attention mask
                        inputs[key] = inputs[key][:, :, :max_length, :max_length]
                    else:
                        # For 2D tensors
                        inputs[key] = inputs[key][:, :max_length]
        
        # Store tensors
        input_ids.append(inputs["input_ids"][0])  # Remove batch dimension
        labels.append(inputs["input_ids"][0])  # Labels are same as input_ids for causal LM
        attention_masks.append(inputs["attention_mask"][0])  # Remove batch dimension
        position_ids.append(inputs["position_ids"][0])  # Remove batch dimension
        
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        position_ids=position_ids,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ) -> None:
        super(SupervisedDataset, self).__init__()
        print("Loading data...")
        sources = utils.jsonl_load(data_path)

        print("Tokenizing inputs... This may take some time...")
        print(f"Creating dataset with custom position ids and attention mask")
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.position_ids = data_dict["position_ids"]
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(self.attention_mask[0].dtype, self.attention_mask[0].shape, self.position_ids[0].dtype, self.position_ids[0].shape, self.input_ids[0].dtype, self.input_ids[0].shape)
        

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, Callable, Dict[str, Any]]]:
        return {
                "input_ids": self.input_ids[i].long(),
                "labels": self.labels[i].long(),
                "attention_mask": self.attention_mask[i].long(), # NOTE: converting attention_mask and position_ids to long as well
                "position_ids": self.position_ids[i].long(),
            }


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.max_length = 64  # Add max_length parameter

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        # Validate shapes before padding
        for i, (ids, mask, pos) in enumerate(zip(input_ids, attention_masks, position_ids)):
            if ids.dim() != 1:
                raise ValueError(f"Expected 1D input_ids, got {ids.dim()}D at index {i}")
            if mask.dim() != 3:
                raise ValueError(f"Expected 3D attention_mask, got {mask.dim()}D at index {i}")
            if pos.dim() != 1:
                raise ValueError(f"Expected 1D position_ids, got {pos.dim()}D at index {i}")
            
            # Truncate if necessary
            if ids.size(0) > self.max_length:
                input_ids[i] = ids[:self.max_length]
                labels[i] = labels[i][:self.max_length]
                attention_masks[i] = mask[:, :self.max_length, :self.max_length]
                position_ids[i] = pos[:self.max_length]

        # Ensure consistent dtypes during padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).long()
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        ).long()

        # Pad attention masks (need to handle 2D case)
        max_len = input_ids.size(1)
        batched_attention_mask = torch.zeros(
            (len(instances), 1, max_len, max_len), 
            dtype=torch.long,
            device=input_ids.device
        )
        
        # Careful padding of attention mask
        for i, mask in enumerate(attention_masks):
            seq_len = mask.size(-1)
            if seq_len > max_len:
                mask = mask[:, :seq_len, :seq_len]
            batched_attention_mask[i, :, :seq_len, :seq_len] = mask

        # Pad position ids with proper masking value
        position_ids = torch.nn.utils.rnn.pad_sequence(
            position_ids, batch_first=True, padding_value=0
        ).long()

        # Ensure attention mask values are exactly 0 or 1
        batched_attention_mask = batched_attention_mask.bool().long()
        
        # Convert attention mask to float for better numerical stability
        batched_attention_mask = batched_attention_mask.to(torch.float32)
        
        # Final validation
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": batched_attention_mask,
            "position_ids": position_ids,
        }

        # Validate final batch
        for k, v in batch.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN values found in {k}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf values found in {k}")

        return batch


class SubsetDatasetWithAttrs(torch.utils.data.Subset):
    """Subset that preserves dataset attributes."""
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        assert 'attention_mask' in item, "Attention mask must be present in item"
        assert 'position_ids' in item, "Position IDs must be present in item"
        assert 'input_ids' in item, "Input IDs must be present in item"
        assert 'labels' in item, "Labels must be present in item"
        item['attention_mask'] = self.dataset.attention_mask[idx]
        item['position_ids'] = self.dataset.position_ids[idx]
        item['input_ids'] = self.dataset.input_ids[idx]
        item['labels'] = self.dataset.labels[idx]
        return item


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    model: transformers.PreTrainedModel,
) -> Dict[str, Union[Dataset, DataCollatorForSupervisedDataset]]:
    """Make dataset and collator for supervised fine-tuning."""
    full_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        model=model,
    )
    
    # NOTE: reduced dataset size for quick testing of train.py script
    # TEMPORARY: Use only a tiny subset of data for quick testing
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"Full dataset length: {len(full_dataset)}")
    total_size = len(full_dataset)
    #total_size = min(len(full_dataset), 1000)  # Only use 100 examples
    #indices = torch.randperm(len(full_dataset))[:total_size]
    #full_dataset = SubsetDatasetWithAttrs(full_dataset, indices)
    
    # Calculate split sizes
    #total_size = len(full_dataset)
    val_size = max(int(0.1 * total_size), 1)  # Ensure at least 1 validation example
    train_size = total_size - val_size
    
    # Create indices for split
    indices = torch.randperm(len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets that preserve attributes
    train_dataset = SubsetDatasetWithAttrs(full_dataset, train_indices)
    eval_dataset = SubsetDatasetWithAttrs(full_dataset, val_indices)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Add gradient clipping
    training_args.max_grad_norm = 1.0
    
    # Force float32 for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Changed from bfloat16 to float32
    print(f"Using device: {device}, dtype: {dtype}")
    
    model, tokenizer = load_model(model_args.model_name_or_path, device, dtype)
    
    # Ensure model is in float32
    model = model.float()
    
    # Initialize weights with small values if needed
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.01)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Verify model dtype
    print(f"Model dtype after initialization: {next(model.parameters()).dtype}")
    
    # Modify model to accept 2D attention masks
    model = order_independent_llm.input_processing.get_2D_attention_accepting_model(model)
    # Modify training arguments based on device
    if device.type == "cpu" or (device.type == "cuda" and torch.cuda.device_count() <= 1):
        print(f"Disabling FSDP for CPU/single-GPU training")
        os.environ["ACCELERATE_USE_FSDP"] = "false"
        training_args.fsdp = ""
        training_args.fsdp_config = None
    '''
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
        )
        model = model.to(device)
        
        # Ensure model parameters are on correct device and dtype

            
        for name, param in model.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
                
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    model = override_forwards(model)
    
    # Add error handling for tokenizer loading
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
    '''
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

    print("Saving initial model weights...")
    # Save initial model weights if not already saved
    try:
        initial_weights_dir = os.path.join(training_args.output_dir, "initial_weights")
        os.makedirs(initial_weights_dir, exist_ok=True)
        if not any(os.scandir(initial_weights_dir)):  # Check if directory is empty
            model.save_pretrained(initial_weights_dir)
            tokenizer.save_pretrained(initial_weights_dir)
            print(f"Saved initial model weights to {initial_weights_dir}")
    except Exception as e:
        print(f"Failed to save initial weights: {e}")
        raise
    else:
        print(f"Initial weights directory already exists at {initial_weights_dir}")
    
    # Create data module with model
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model=model,
    )
    try:
        training_args.remove_unused_columns = False
    except:
        print("Failed to set remove_unused_columns to False in training_args")
        pass
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
    print("attention_mask:", first_batch["attention_mask"])
    print("position_ids:", first_batch["position_ids"])
    print(first_batch["input_ids"].unsqueeze(0).shape, first_batch["labels"].unsqueeze(0).shape, first_batch["position_ids"].unsqueeze(0).shape, first_batch["attention_mask"].unsqueeze(0).shape)
    # Then run a quick forward pass manually
    with torch.set_grad_enabled(True):
        outputs = model(
            input_ids=first_batch["input_ids"].unsqueeze(0).to(device),
            labels=first_batch["labels"].unsqueeze(0).to(device),
            position_ids=first_batch["position_ids"].unsqueeze(0).to(device),
            attention_mask=first_batch["attention_mask"].unsqueeze(0).to(device),
        )
        print(outputs.keys())
        print(outputs.loss, outputs.loss.requires_grad, outputs.logits.shape)
        print(outputs.logits)
    print(f"Finish debug sample batch")

    # Add timing callback
    class TimingCallback(transformers.TrainerCallback):
        def __init__(self, print_interval_steps=100):
            self.print_interval_steps = print_interval_steps
            self.start_time = None
            self.last_print_time = None
            self.last_print_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
            print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total epochs: {args.num_train_epochs}")
            
             # Evaluate before training
            print("\nInitial model evaluation:")
            initial_metrics = trainer.evaluate()
            print(f"Initial eval_loss: {initial_metrics['eval_loss']:.4f}")
            # Get initial training loss
            print("\nCalculating initial training loss:")
            initial_train_metrics = trainer.evaluate(eval_dataset=data_module["train_dataset"], metric_key_prefix="train")
            print(f"Initial train_loss: {initial_train_metrics['train_loss']:.4f}")

        def on_step_end(self, args, state, control, **kwargs):
            if self.last_print_time is None:
                self.last_print_time = self.start_time
                self.last_print_step = state.global_step
                return

            current_step = state.global_step
            if current_step - self.last_print_step >= self.print_interval_steps:
                current_time = time.time()
                steps_per_second = (current_step - self.last_print_step) / (current_time - self.last_print_time)
                
                # Avoid division by zero
                if steps_per_second > 0:
                    remaining_steps = state.max_steps - current_step
                    remaining_time = remaining_steps / steps_per_second
                    
                    # Calculate current epoch and progress
                    steps_per_epoch = state.max_steps / args.num_train_epochs
                    current_epoch = (current_step / steps_per_epoch) + 1
                    epoch_step = current_step % steps_per_epoch
                    
                    # Calculate completion time
                    completion_time = datetime.now() + timedelta(seconds=remaining_time)
                    
                    print(f"\nTraining Progress:")
                    print(f"Epoch: {current_epoch:.2f}/{args.num_train_epochs} "
                          f"(Step {epoch_step:.0f}/{steps_per_epoch:.0f} in current epoch)")
                    print(f"Global Step: {current_step}/{state.max_steps}")
                    print(f"Speed: {steps_per_second:.2f} steps/second")
                    print(f"Estimated time remaining: {timedelta(seconds=int(remaining_time))}")
                    print(f"Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    self.last_print_time = current_time
                    self.last_print_step = current_step

        def on_train_end(self, args, state, control, **kwargs):
            total_time = time.time() - self.start_time
            print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total training time: {timedelta(seconds=int(total_time))}")

    # Add callback to trainer
    trainer.add_callback(TimingCallback(print_interval_steps=100))
    
    print(f"Training model...")
    trainer.create_optimizer()
    print("Optimizer defaults: ", trainer.optimizer.defaults)
    trainer.train()

    # Wrap saving in no_grad context
    with torch.no_grad():
        print("Saving model state...")
        trainer.save_state()
        
        print("Saving final model weights...")
        final_weights_dir = os.path.join(training_args.output_dir, "final_weights")
        os.makedirs(final_weights_dir, exist_ok=True)
        trainer.save_model(output_dir=final_weights_dir)
        print(f"Saved final model weights to {final_weights_dir}")


def get_pca_components(model_name: str) -> int:
    if model_name not in utils.N_PCA_COMPONENTS_DICT:
        raise ValueError(f"Model {model_name} not found in utils.N_PCA_COMPONENTS_DICT. "
                       f"Available models: {list(utils.N_PCA_COMPONENTS_DICT.keys())}")
    return utils.N_PCA_COMPONENTS_DICT[model_name]


def get_parallel_inputs(
    prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Process a prompt and return the input tensors needed for processing.
    Handles both regular prompts and those with parallel substrings.
    """
    # Check if this is a parallel processing prompt
    if "<|start_2d|>" not in prompt:
        # Regular prompt without parallel sections
        tokAll = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            return_token_type_ids=False,
        )
        nTokAll = tokAll["input_ids"].size(1)
        
        # Create standard causal attention mask expanded to 4D
        causal_mask = torch.tril(torch.ones((nTokAll, nTokAll), dtype=torch.bool))
        attention_mask_2d = causal_mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
        attention_mask_2d = attention_mask_2d.to(torch.float32)  # Convert to float32 like parallel case
        
        # Create standard ascending position IDs
        position_ids = torch.arange(0, nTokAll).unsqueeze(0)  # Shape: [1, seq_len]
        
        return {
            "input_ids": tokAll["input_ids"],
            "position_ids": position_ids,
            "attention_mask": attention_mask_2d
        }

    # Original parallel processing code for prompts with <|start_2d|> tags
    if "<|start_2d|>" not in prompt or "<|end_2d|>" not in prompt:
        raise ValueError("Prompt must contain <|start_2d|> and <|end_2d|> tags")
        
    prefix, rest = prompt.split("<|start_2d|>")
    parallel_part, suffix = rest.split("<|end_2d|>")
    parallel_substrings = parallel_part.split("<|split_2d|>")
    
    # Tokenize each part
    tokA = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
        return_token_type_ids=False,
    )
    
    tokD = tokenizer(
        suffix,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    )
    
    tokParallel = [
        tokenizer(text, return_tensors="pt", add_special_tokens=False)
        for text in parallel_substrings
    ]
    
    # Combine tokens into single sequence
    tokAll = order_independent_llm.input_processing.get_tokenized_input_prompt(tokA, tokParallel, tokD)
    
    # Generate position IDs for parallel processing
    position_ids, _, _ = order_independent_llm.input_processing.get_position_ids_nopad_n_options(
        tokA, tokParallel, tokD, tokenizer=tokenizer
    )
    
    # Generate 2D attention mask
    attention_mask_2d = order_independent_llm.input_processing.get_attention_mask_2d_n_options(
        tokA, tokParallel, tokD, tokAll
    )
    
    return {
        "input_ids": tokAll["input_ids"],
        "position_ids": position_ids,
        "attention_mask": attention_mask_2d
    }

from transformers.trainer import *
from transformers.trainer import _is_peft_model
class TrustedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for each example individually.
        If any example produces NaN/Inf, print that example's input info.
        Otherwise, aggregate all losses.
        """
        print("\nDEBUG INPUT SHAPES (BATCH):")
        for k, v in inputs.items():
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            print(f"Sample values: min={v.min().item()}, max={v.max().item()}")

        # Ensure consistent device placement and dtype
        device = model.device
        model_dtype = next(model.parameters()).dtype
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Convert attention mask to model's dtype if it mismatches
        if inputs["attention_mask"].dtype != model_dtype:
            inputs["attention_mask"] = inputs["attention_mask"].to(model_dtype)

        batch_size = inputs["input_ids"].shape[0]

        # We'll accumulate valid losses here
        example_losses = []

        for i in range(batch_size):
            single_inputs = {}
            for k, v in inputs.items():
                # For attention_mask or others that may have extra dimensions, slice carefully
                if v.dim() > 1:
                    # We want just the i-th sample: keep the batch dimension for forward pass
                    single_inputs[k] = v[i : i + 1]
                else:
                    single_inputs[k] = v[i]

            try:
                with torch.autograd.detect_anomaly():
                    # Forward pass for a single example
                    outputs = model(**single_inputs)
                    loss = outputs[0] if not isinstance(outputs, dict) else outputs["loss"]
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("WARNING: Detected NaN/Inf loss for example index:", i)
                        print("Model dtype:", model_dtype)
                        print("Loss dtype:", loss.dtype)
                        # Print out the single_inputs stats for debugging
                        for k, v in single_inputs.items():
                            # shape, device, min/max etc.
                            print(f"Example {i} | {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                            print(f"  values range: min={v.min().item()}, max={v.max().item()}")
                            print(single_inputs["input_ids"])
                            print(single_inputs["labels"])
                            print(single_inputs["position_ids"])
                            print(single_inputs["attention_mask"])
                        raise ValueError(f"NaN/Inf loss in example index {i}")
                    else:
                        example_losses.append(loss)
            except Exception as e:
                print(f"Error in compute_loss for example index {i}: {e}")
                raise

        # If no example produced NaN/Inf, aggregate the losses
        if not example_losses:
            # In case the entire batch was empty or something unexpected
            raise ValueError("No valid examples in batch (all produced NaN/Inf or batch was empty)")

        # Compute average
        total_loss = torch.stack(example_losses).mean()

        return (total_loss, outputs) if return_outputs else total_loss
    
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        # Don't remove any columns - skip the _remove_unused_columns call
        # that normally happens in Trainer
        return data_collator
    
    
    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        """Override to keep all columns"""
        return dataset

if __name__ == "__main__":
    # Configure logging
    #logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    level=print,
    #)
    train()
