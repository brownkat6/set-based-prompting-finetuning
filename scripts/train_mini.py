import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import Trainer, HfArgumentParser
import order_independent_llm
from order_independent_llm.input_processing import load_model

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)

def main():
    # Force CPU
    device = torch.device("cpu")
    
    # Parse args
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model(model_args.model_name_or_path, device, torch.float32)
    
    print("\nPreparing datasets...")
    from train_full import make_supervised_data_module
    data_module = make_supervised_data_module(tokenizer, data_args, model)
    
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )
    
    print("\nChecking initial evaluation...")
    # Get initial eval loss
    initial_metrics = trainer.evaluate()
    print(f"Initial eval_loss: {initial_metrics.get('eval_loss', 'NOT FOUND')}")
    
    # Get initial training loss
    print("\nChecking initial training loss...")
    initial_train_metrics = trainer.evaluate(
        eval_dataset=data_module["train_dataset"], 
        metric_key_prefix="train"
    )
    print(f"Initial train_loss: {initial_train_metrics.get('train_loss', 'NOT FOUND')}")
    
    # Debug: check a single batch
    print("\nChecking single batch forward pass...")
    first_batch = next(iter(data_module["train_dataset"]))
    outputs = model(
        input_ids=first_batch["input_ids"].unsqueeze(0),
        labels=first_batch["labels"].unsqueeze(0),
        attention_mask=first_batch["attention_mask"].unsqueeze(0),
        position_ids=first_batch["position_ids"].unsqueeze(0)
    )
    print(f"Single batch loss: {outputs.loss.item()}")

if __name__ == "__main__":
    main() 