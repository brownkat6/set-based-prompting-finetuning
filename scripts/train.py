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

def get_parallel_inputs(
    prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Process a prompt with parallel substrings and return the input tensors needed for parallel processing.
    
    Args:
        prompt: String in format "prefix<|start_2d|>str1<|split_2d|>str2<|end_2d|>suffix"
        tokenizer: The tokenizer to use for processing
    
    Returns:
        Dictionary containing:
            - input_ids: tensor of token ids
            - position_ids: tensor of position ids for parallel processing
            - attention_mask: 2D attention mask for parallel processing
    """
    # Split the prompt into prefix, parallel substrings, and suffix
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
    position_ids = order_independent_llm.input_processing.get_position_ids_nopad_n_options(
        tokA, tokParallel, tokD, tokenizer=tokenizer
    )[0]
    
    # Generate 2D attention mask
    attention_mask_2d = order_independent_llm.input_processing.get_attention_mask_2d_n_options(
        tokA, tokParallel, tokD, tokAll
    )
    
    return {
        "input_ids": tokAll["input_ids"],
        "position_ids": position_ids,
        "attention_mask": attention_mask_2d
    }

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

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
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
        super().__post_init__()
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
    is_order_independent: bool = True,
) -> Dict[str, List[Union[torch.Tensor, Dict[str, Any]]]]:
    """Preprocess the data by tokenizing."""
    input_ids = []
    position_ids = []
    attention_masks_2d = []
    
    for source in sources:
        if is_order_independent:
            # Process with parallel input handling
            parallel_inputs = get_parallel_inputs(source, tokenizer)
            input_ids.append(parallel_inputs["input_ids"])
            position_ids.append(parallel_inputs["position_ids"])
            attention_masks_2d.append(parallel_inputs["attention_mask"])
        else:
            # Standard processing
            tokenized = tokenizer(
                source,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            input_ids.append(tokenized.input_ids[0])
    
    labels = copy.deepcopy(input_ids)
    result = {"input_ids": input_ids, "labels": labels}
    
    if is_order_independent:
        result.update({
            "position_ids": position_ids,
            "attention_mask": attention_masks_2d
        })
        
    return result

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        is_order_independent: bool = True
    ) -> None:
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        sources = utils.jsonl_load(data_path)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, tokenizer, is_order_independent)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.is_order_independent = is_order_independent
        if is_order_independent:
            self.position_ids = data_dict["position_ids"]
            self.attention_mask_2d = data_dict["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        result = {
            "input_ids": self.input_ids[i].long(),
            "labels": self.labels[i].long(),
        }
        if self.is_order_independent:
            result.update({
                "position_ids": self.position_ids[i],
                "attention_mask": self.attention_mask_2d[i]
            })
        return result

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Collect lists of input_ids / labels
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]

        # Flatten to 1D if needed, e.g. if shape is [1, seq_len] -> [seq_len]
        def maybe_flatten(t: torch.Tensor) -> torch.Tensor:
            # If dimension 0 is 1 and dimension 1 is seq_len, flatten
            if t.dim() == 2 and t.size(0) == 1:
                return t.view(-1)
            return t

        input_ids_list = [maybe_flatten(x) for x in input_ids_list]
        labels_list = [maybe_flatten(x) for x in labels_list]

        # Truncate to model max length, then gather the max length for this batch
        max_len_for_model = self.tokenizer.model_max_length
        input_ids_list = [x[:max_len_for_model] for x in input_ids_list]
        labels_list = [x[:max_len_for_model] for x in labels_list]

        # Now pad them to the same length in this batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # Construct the final dict
        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }

        # Now handle attention_mask or position_ids if they exist
        # If your dataset doesn't store them, skip. Otherwise flatten/truncate similarly.
        if "attention_mask" in instances[0]:
            attn_list = [inst["attention_mask"] for inst in instances]
            # Example: if shape is [1, seq_len, seq_len], flatten or fix dims
            def process_2d_mask(m: torch.Tensor) -> torch.Tensor:
                # If needed, clip to max_len_for_model along last two dims
                seq_len = m.size(-1)
                if seq_len > max_len_for_model:
                    m = m[..., :max_len_for_model, :max_len_for_model]
                return m

            attn_list = [process_2d_mask(x) for x in attn_list]
            # Now find the largest seq length in this batch after clipping
            max_len_in_batch = max(x.size(-1) for x in attn_list)

            # Pad each mask to that max_len_in_batch in the last two dims
            padded_masks = []
            for am in attn_list:
                pad_amt = max_len_in_batch - am.size(-1)
                if pad_amt > 0:
                    am = torch.nn.functional.pad(am, (0, pad_amt, 0, pad_amt), value=0)
                padded_masks.append(am)
            attention_mask = torch.stack(padded_masks, dim=0)  # shape: (batch, 1, L, L)
            batch["attention_mask"] = attention_mask

        if "position_ids" in instances[0]:
            pos_list = [inst["position_ids"] for inst in instances]
            # Flatten if needed
            pos_list = [maybe_flatten(x) for x in pos_list]
            pos_list = [x[:max_len_for_model] for x in pos_list]
            position_ids = torch.nn.utils.rnn.pad_sequence(
                pos_list,
                batch_first=True,
                padding_value=0
            )
            batch["position_ids"] = position_ids

        return batch

class SubsetDatasetWithAttrs(torch.utils.data.Subset):
    """Subset that preserves dataset attributes."""
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['attention_mask'] = self.dataset.attention_mask[idx]
        item['position_ids'] = self.dataset.position_ids[idx]
        return item

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    is_order_independent: bool = True
) -> Dict[str, Union[Dataset, DataCollatorForSupervisedDataset]]:
    """Make dataset and collator for supervised fine-tuning."""
    full_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        is_order_independent=is_order_independent
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"Full dataset length: {len(full_dataset)}")
    total_size = min(len(full_dataset), 1000)  # Only use 1000 examples
    indices = torch.randperm(len(full_dataset))[:total_size]
    full_dataset = SubsetDatasetWithAttrs(full_dataset, indices)
    
    # Calculate split sizes
    val_size = max(int(0.1 * total_size), 1)
    train_size = total_size - val_size
    
    # Create indices for split
    indices = torch.randperm(len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
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
        
        # Verify LoRA modules
        #print("\nVerifying LoRA modules:")
        #for name, module in model.named_modules():
        #    if 'lora_' in name:
        #        print(f"Found LoRA module: {name}")
                
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
        is_order_independent=model_args.is_order_independent
    )

    trainer = Trainer(
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
