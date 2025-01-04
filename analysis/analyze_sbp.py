import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import order_independent_llm
import order_independent_llm.plot_helpers
import json

# Base paths
BASE_DIR = "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp"
RESULTS_DIR_MMLU = "results/mmlu_quoted_permutations"
RESULTS_DIR_CSQA = "results/csqa_quoted_permutations"

# Define model pairs (pre and post finetuning)
model_pairs = [
    {
        "name": "gpt2",
        "pre_path": "gpt2",
        "post_path": os.path.join(BASE_DIR, "gpt2", "latest")  # assuming latest is symlink to most recent run
    },
    {
        "name": "Llama-2-7b",
        "pre_path": "meta-llama/Llama-2-7b-hf",
        "post_path": os.path.join(BASE_DIR, "meta-llama_Llama-2-7b-hf", "latest")
    }
]

def load_results(model_path, dataset_type):
    """Load results for a given model and dataset type (MMLU or CSQA)"""
    results_dir = RESULTS_DIR_MMLU if dataset_type == "MMLU" else RESULTS_DIR_CSQA
    pattern = os.path.join("..", results_dir, f"*{model_path}*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results found for {model_path} in {results_dir}")
        return None
        
    return pd.concat([order_independent_llm.load_to_dataframe(f, fail_on_empty=True) for f in files])

def create_permutation_boxplot(model_results, dataset_type, output_name):
    """Create boxplot showing distribution of accuracies across permutations for all models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for boxplot
    box_data = []
    
    # Process each model's data
    for model_info in model_results:
        pre_data = model_info['pre_data']
        post_data = model_info['post_data']
        model_name = model_info['name']
        
        if pre_data is None:
            continue
            
        # Get permutation columns
        perm_cols = [col for col in pre_data.columns if col.startswith('normal_permuted_')]
        
        # Add pre-finetuning permutation data
        for _, row in pre_data.iterrows():
            for col in perm_cols:
                box_data.append({
                    'Accuracy': row[col],
                    'Type': 'Permutations',
                    'Model': model_name
                })
        
        # Add pre-finetuning order independent
        pre_oid = pre_data[pre_data['response_type'] == 'order_independent']['is_correct_answer'].mean()
        box_data.append({
            'Accuracy': pre_oid,
            'Type': 'Pre-OID',
            'Model': model_name
        })
        
        # Add post-finetuning order independent
        if post_data is not None:
            post_oid = post_data[post_data['response_type'] == 'order_independent']['is_correct_answer'].mean()
            box_data.append({
                'Accuracy': post_oid,
                'Type': 'Post-OID',
                'Model': model_name
            })
    
    # Create plot
    box_df = pd.DataFrame(box_data)
    
    # Create grouped boxplot
    sns.boxplot(
        data=box_df,
        x='Model',
        y='Accuracy',
        hue='Type',
        ax=ax
    )
    
    ax.set_title(f"{dataset_type} Accuracy Distribution - All Models")
    ax.set_xlabel("")
    ax.set_ylabel("Top 1 Accuracy")
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"plots/{output_name}_boxplot.png", bbox_inches='tight')
    plt.close()

def create_comparison_barplot(model_results, dataset_type, output_name):
    """Create barplot comparing normal vs order independent accuracies for all models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_data = []
    
    # Process each model's data
    for model_info in model_results:
        pre_data = model_info['pre_data']
        post_data = model_info['post_data']
        model_name = model_info['name']
        
        if pre_data is None:
            continue
            
        # Pre-finetuning normal accuracy with error bars
        normal_acc = pre_data[pre_data['response_type'] == 'normal']['is_correct_answer'].mean()
        normal_rev_acc = pre_data[pre_data['response_type'] == 'normal_reversed']['is_correct_answer'].mean()
        error = abs(normal_acc - normal_rev_acc)
        
        plot_data.append({
            'Model': model_name,
            'Type': 'Normal',
            'Accuracy': normal_acc,
            'Error': error
        })
        
        # Pre-finetuning order independent
        pre_oid = pre_data[pre_data['response_type'] == 'order_independent']['is_correct_answer'].mean()
        plot_data.append({
            'Model': model_name,
            'Type': 'Pre-OID',
            'Accuracy': pre_oid,
            'Error': 0
        })
        
        # Post-finetuning order independent
        if post_data is not None:
            post_oid = post_data[post_data['response_type'] == 'order_independent']['is_correct_answer'].mean()
            plot_data.append({
                'Model': model_name,
                'Type': 'Post-OID',
                'Accuracy': post_oid,
                'Error': 0
            })
    
    # Create plot
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped barplot
    sns.barplot(
        data=plot_df,
        x='Model',
        y='Accuracy',
        hue='Type',
        ax=ax
    )
    
    # Add error bars
    for i, model in enumerate(plot_df['Model'].unique()):
        model_data = plot_df[plot_df['Model'] == model]
        normal_data = model_data[model_data['Type'] == 'Normal']
        if not normal_data.empty:
            ax.errorbar(
                x=i - 0.2,  # Adjust position to align with correct bar
                y=normal_data['Accuracy'].iloc[0],
                yerr=normal_data['Error'].iloc[0],
                fmt='none',
                c='black'
            )
    
    ax.set_title(f"{dataset_type} Accuracy Comparison - All Models")
    ax.set_xlabel("")
    ax.set_ylabel("Top 1 Accuracy")
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"plots/{output_name}_barplot.png", bbox_inches='tight')
    plt.close()

def load_perplexity_results(model_dir):
    """Load WikiText-103 perplexity results from benchmarks.jsonl"""
    benchmarks_file = os.path.join(model_dir, "benchmarks.jsonl")
    if not os.path.exists(benchmarks_file):
        print(f"No benchmarks file found at {benchmarks_file}")
        return None
        
    results = []
    with open(benchmarks_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Extract initial and final weights perplexity
    initial_perp = df[df['model_path'].str.contains('initial_weights')]['wikitext_perplexity'].iloc[0]
    final_perp = df[df['model_path'].str.contains('final_weights')]['wikitext_perplexity'].iloc[0]
    
    return {
        'initial_perplexity': initial_perp,
        'final_perplexity': final_perp
    }

def create_perplexity_plot(perplexity_data, dataset_type, output_name):
    """Create barplot comparing initial and final perplexities across models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = []
    for model_name, perp in perplexity_data.items():
        if perp is not None:
            plot_data.extend([
                {
                    'Model': model_name,
                    'Type': 'Initial Weights',
                    'Perplexity': perp['initial_perplexity']
                },
                {
                    'Model': model_name,
                    'Type': 'Final Weights',
                    'Perplexity': perp['final_perplexity']
                }
            ])
    
    if not plot_data:
        print(f"No perplexity data available for {dataset_type}")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped barplot
    sns.barplot(
        data=plot_df,
        x='Model',
        y='Perplexity',
        hue='Type',
        ax=ax
    )
    
    ax.set_title(f"WikiText-103 Perplexity - {dataset_type} Finetuning")
    ax.set_xlabel("")
    ax.set_ylabel("Perplexity")
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"plots/{output_name}_perplexity.png", bbox_inches='tight')
    plt.close()

def analyze_dataset(dataset_type):
    """Analyze results for either MMLU or CSQA"""
    model_results = []
    
    # Collect all model results first
    for pair in model_pairs:
        # Load pre-finetuning results
        pre_results = load_results(pair["pre_path"], dataset_type)
        if pre_results is None:
            continue
            
        # Load post-finetuning results
        post_results = load_results(pair["post_path"], dataset_type)
        
        model_results.append({
            'name': pair['name'],
            'pre_data': pre_results,
            'post_data': post_results
        })
    
    # Create plots with all models
    create_permutation_boxplot(
        model_results,
        dataset_type,
        f"{dataset_type.lower()}_all_models"
    )
    create_comparison_barplot(
        model_results,
        dataset_type,
        f"{dataset_type.lower()}_all_models"
    )
    
    # Create perplexity plot (unchanged)
    perplexity_data = {}
    for pair in model_pairs:
        perp_results = load_perplexity_results(pair["post_path"])
        if perp_results is not None:
            perplexity_data[pair["name"]] = perp_results
    
    create_perplexity_plot(
        perplexity_data,
        dataset_type,
        f"{dataset_type.lower()}_models"
    )

def main():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Analyze both datasets
    analyze_dataset("MMLU")
    analyze_dataset("CSQA")

if __name__ == "__main__":
    main() 