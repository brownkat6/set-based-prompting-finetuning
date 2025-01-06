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


def _tokenize_fn(
    strings: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict[str, List[Union[torch.Tensor, int]]]:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, List[Union[torch.Tensor, Dict[str, Any]]]]:
    """Preprocess the data by tokenizing and generating parallel processing tensors."""
    input_ids = []
    labels = []
    attention_masks = []
    position_ids = []
    
    for source in sources:
        # Get parallel processing tensors
        inputs = get_parallel_inputs(source, tokenizer)
        
        # Store tensors
        input_ids.append(inputs["input_ids"][0])  # Remove batch dimension
        labels.append(inputs["input_ids"][0])  # Labels are same as input_ids for causal LM
        attention_masks.append(inputs["attention_mask"][0])  # Remove batch dimension
        position_ids.append(inputs["position_ids"][0])  # Remove batch dimension
    
    # print some info about the dataset we just generated
    print(f"Generated dataset with {len(input_ids)} examples")
    print(sources[:5])
    print(f"First example: {input_ids[0]}")
    print(f"First label: {labels[0]}")
    print(f"First attention mask: {attention_masks[0]}")
    print(f"First position ids: {position_ids[0]}")
    
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
        logging.warning("Loading data...")
        sources = utils.jsonl_load(data_path)

        logging.warning("Tokenizing inputs... This may take some time...")
        print(f"Creating dataset with custom position ids and attention mask")
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.position_ids = data_dict["position_ids"]
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, Callable, Dict[str, Any]]]:
        return {
                "input_ids": self.input_ids[i].long(),
                "labels": self.labels[i].long(),
                "attention_mask": self.attention_mask[i],
                "position_ids": self.position_ids[i]
            }


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        # Ensure consistent dtype during padding
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
            dtype=torch.long
        )
        for i, mask in enumerate(attention_masks):
            seq_len = mask.size(-1)
            batched_attention_mask[i, :, :seq_len, :seq_len] = mask

        # Pad position ids
        position_ids = torch.nn.utils.rnn.pad_sequence(
            position_ids, batch_first=True, padding_value=0
        ).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": batched_attention_mask,
            "position_ids": position_ids,
        }


class SubsetDatasetWithAttrs(torch.utils.data.Subset):
    """Subset that preserves dataset attributes."""
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Preserve noise_fn and noise_fn_kwargs if they exist
        #if hasattr(self.dataset, 'noise_fn'):
        #    item['noise_fn'] = self.dataset.noise_fn
        #if hasattr(self.dataset, 'noise_fn_kwargs'):
        #    item['noise_fn_kwargs'] = self.dataset.noise_fn_kwargs[idx]
        item['attention_mask'] = self.dataset.attention_mask[idx]
        item['position_ids'] = self.dataset.position_ids[idx]
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

class TrustedTrainer(Trainer):
    """Custom trainer that handles noise functions during both training and evaluation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Run evaluation and returns metrics."""
        print("Evaluating...")
        
        # Get current epoch number (1-based indexing)
        current_epoch = int(self.state.epoch)
        print(f"\nEvaluating epoch {current_epoch}")
        
        # Print eval dataset length
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        print(f"Evaluation dataset length: {len(eval_dataset)}")
        
        # Get dataloader
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        print(f"Number of evaluation batches: {len(eval_dataloader)}")
        
        # Initialize metrics
        total_loss = 0.0
        num_batches = 0
        
        # Evaluation loop
        model = self.model
        model.eval()
        
        for batch in eval_dataloader:
            with torch.no_grad():
                loss = self.evaluation_step(model, batch)
                total_loss += loss.item()
                print(f"Batch loss: {loss}")
                num_batches += 1
                
        # Compute average loss
        if num_batches == 0:
            print("No batches found")
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        
        metrics = {
            "eval_loss": avg_loss,
            "epoch": current_epoch
        }
        
        print(f"Evaluation metrics: {metrics}")
        return metrics

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs by placing them on the correct device and dtype."""
        inputs = super()._prepare_inputs(inputs)
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k not in ["input_ids", "labels", "attention_mask", "position_ids"]:
                    inputs[k] = v.to(self.dtype)
        
        return inputs

    def evaluation_step(self, model, inputs):
        """Single evaluation step."""
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            
            # Validate tensor shapes
            batch_size = inputs["input_ids"].size(0)
            seq_len = inputs["input_ids"].size(1)
            
            assert inputs["attention_mask"].size() == (batch_size, 1, seq_len, seq_len), \
                f"Attention mask should be 4D with shape {(batch_size, 1, seq_len, seq_len)}, got {inputs['attention_mask'].size()}"
            
            assert inputs["position_ids"].size() == (batch_size, seq_len), \
                f"Position ids should be 2D with shape {(batch_size, seq_len)}, got {inputs['position_ids'].size()}"
            
            loss = self.compute_loss(model, inputs)

        return loss

    def get_eval_dataloader(self, eval_dataset=None):
        """Override to ensure we use our custom data collator for evaluation."""
        print("Getting eval dataloader...")
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs):
        """Training step with shape validation."""
        #print(f"training step")
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Validate tensor shapes
        batch_size = inputs["input_ids"].size(0)
        seq_len = inputs["input_ids"].size(1)
        
        assert inputs["attention_mask"].size() == (batch_size, 1, seq_len, seq_len), \
            f"Attention mask should be 4D with shape {(batch_size, 1, seq_len, seq_len)}, got {inputs['attention_mask'].size()}"
        
        assert inputs["position_ids"].size() == (batch_size, seq_len), \
            f"Position ids should be 2D with shape {(batch_size, seq_len)}, got {inputs['position_ids'].size()}"
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        loss.backward()
        
        l = loss.detach()
        # check if loss is nan
        if torch.isnan(l):
            print("Training loss is nan")
            raise ValueError("Training step loss is nan")
        return l

    def create_optimizer(self):
        """Setup optimizer with a dimension-based grouping so RMSNorm is included in no_decay."""
        if self.optimizer is None:
            print("Creating optimizer...")

            # 1. Separate params into 'decay' vs. 'no_decay' via dimension (and bias).
            decay, no_decay = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # Put 1D parameters (e.g., LayerNorm, RMSNorm, embeddings) and biases in no_decay
                if param.ndim == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            optimizer_grouped_parameters = [
                {
                    "params": decay,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": no_decay,
                    "weight_decay": 0.0,
                },
            ]

            print("Getting optimizer class and kwargs...")
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # 2. Check FSDP compatibility and other advanced features
            print("Checking for FSDP...")
            try:
                if hasattr(self.model, "is_fsdp_enabled") and self.model.is_fsdp_enabled:
                    print("FSDP detected - applying FSDP-specific optimizations")
                    optimizer_kwargs["fsdp_optimizer"] = True
                    if getattr(self.args, "fp16", False):
                        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                        self.scaler = ShardedGradScaler()
            except Exception as e:
                print(f"Error during FSDP check: {e}")
                print("Continuing with standard optimizer setup")

            print("Creating optimizer with the following config:")
            # NOTE: set foreach=False to avoid FSDP issues
            optimizer_kwargs["foreach"] = False
            print(f"  Optimizer class: {optimizer_cls.__name__}")
            print(f"  Optimizer kwargs: {optimizer_kwargs}")
            print(f"  Number of parameter groups: {len(optimizer_grouped_parameters)}")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            print("Optimizer created successfully")
        else:
            print("Optimizer already exists")
            # set foreach=False to avoid FSDP issues
            self.optimizer.foreach = False

        return self.optimizer

    def _verify_optimizer_state(self):
        """Verify optimizer state matches parameter shapes."""
        total_params = 0
        total_state_size = 0
        
        '''
        for group_idx, group in enumerate(self.optimizer.param_groups):
            print(f"\nParameter Group {group_idx}:")
            for param_idx, p in enumerate(group['params']):
                # Check parameter
                print(f"\nParameter {param_idx}:")
                print(f"  Shape: {p.shape}")
                print(f"  Elements: {p.numel()}")
                total_params += p.numel()
                
                # Print parameter info without checking optimizer state membership
                try:
                    print("  Parameter info:")
                    print(f"    Size: {p.numel()}")
                    if hasattr(p, 'grad') and p.grad is not None:
                        print(f"    Gradient shape: {p.grad.shape}")
                        print(f"    Gradient size: {p.grad.numel()}")
                        total_state_size += p.grad.numel()
                except Exception as e:
                    print(f"  Error getting parameter info: {e}")
        '''

        print(f"\nTotal Statistics:")
        print(f"Total parameter elements: {total_params}")
        print(f"Total gradient elements: {total_state_size}")
        # print whether optimizer is using foreach
        print(f"Optimizer using foreach: {self.optimizer.foreach}")

def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # More robust GPU checking and dtype setting
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
    model, tokenizer = load_model(model_args.model_name_or_path, device, dtype)
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
            logging.info(f"Saved initial model weights to {initial_weights_dir}")
    except Exception as e:
        logging.error(f"Failed to save initial weights: {e}")
        raise
    else:
        logging.info(f"Initial weights directory already exists at {initial_weights_dir}")
    
    # Create data module with model
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model=model,
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
        logging.info(f"Saved final model weights to {final_weights_dir}")


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


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train()
