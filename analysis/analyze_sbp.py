import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import order_independent_llm
import order_independent_llm.plot_helpers
import json
import seaborn as sns
from functools import lru_cache

# Base paths
BASE_DIR = "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp"
RESULTS_DIR_MMLU = "results/mmlu_quoted"
RESULTS_DIR_CSQA = "results/csqa_quoted"
OUTPUT_DIR = "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp/meta-llama/Llama-2-7b-hf/20250105-005619_tags-False"

# Define model pairs (pre and post finetuning)
model_pairs = [
    {
        "name": "Llama-2-7b",
        "pre_path": "meta-llama_Llama-2-7b-hf-50",
        "output_dir": "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp/meta-llama/Llama-2-7b-hf/20250105-005619_tags-False", # stores benchmarks.jsonl
    },
    {
        "name": "Llama-2-7b : QA",
        "pre_path": "meta-llama_Llama-2-7b-hf-50",
        "output_dir": "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp/meta-llama/Llama-2-7b-hf/mmlu_quoted_qa/20250106-022456-False", # stores benchmarks.jsonl
    },
    {
        "name": "Llama-2-7b : QA + Wiki",
        "pre_path": "meta-llama_Llama-2-7b-hf-50",
        "output_dir": "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp/meta-llama/Llama-2-7b-hf/mmlu_quoted_qa_wiki/20250106-022502-False", # stores benchmarks.jsonl
    },
    {
        "name": "Llama-2-7b : QA + Start/End Tags",
        "pre_path": "meta-llama_Llama-2-7b-hf-50",
        "output_dir": "/n/netscratch/dwork_lab/Lab/katrina/finetuning_sbp/meta-llama/Llama-2-7b-hf/mmlu_quoted_qa_s2d/20250106-022514-False", # stores benchmarks.jsonl
    },
]
for i in range(len(model_pairs)):
    model_pairs[i]["post_path_csqa"] = f"results/csqa_quoted/"+model_pairs[i]["output_dir"].replace("/","_")+"_final_weights-50"
    model_pairs[i]["post_path_mmlu"] = f"results/mmlu_quoted/"+model_pairs[i]["output_dir"].replace("/","_")+"_final_weights-50"
    model_pairs[i]["pre_path_csqa"] = f"results/csqa_quoted/"+model_pairs[i]["pre_path"]
    model_pairs[i]["pre_path_mmlu"] = f"results/mmlu_quoted/"+model_pairs[i]["pre_path"]
    model_pairs[i]["pre_path_csqa_perm"] = f"results/csqa_quoted_permutations/"+model_pairs[i]["pre_path"]
    model_pairs[i]["pre_path_mmlu_perm"] = f"results/mmlu_quoted_permutations/"+model_pairs[i]["pre_path"]

@lru_cache
def load_results(model_path):
    """Load results for a given model and dataset type (MMLU or CSQA)"""
    files = glob.glob(f"*{model_path}/*.jsonl")
    
    if not files:
        print(f"No results found for {model_path}")
        return None
        
    return pd.concat([order_independent_llm.load_to_dataframe(f, fail_on_empty=True) for f in files])

def create_permutation_boxplot(model_results, dataset_type, output_name):
    """Create boxplot showing distribution of accuracies across permutations for all models"""
    df = pd.concat([model_info['pre_data_perm'].assign(model=model_info['name']) for model_info in model_results])
    df_post = pd.concat([model_info['post_data'].assign(model=model_info['name']) for model_info in model_results])
    df_post['model'] = "meta-llama/" + df_post['model']
    df['model'] = "meta-llama/" + df['model']
    perms = list(df['response_type'].unique())[3:]
    fig, ax  = plt.subplots(figsize = (7,5))
    sns.boxplot(
        x = 'model',
        y ='is_correct_answer',
        data = df[df['response_type'].isin(perms)][['model','is_correct_answer','response_type']].groupby(['model','response_type']).mean().reset_index(),
        ax = ax,
        whis=[0, 100],
        width =.6,
    )
    sns.stripplot(
        x = 'model',
        y ='is_correct_answer',
        data = df[df['response_type'].isin(perms)][['model','is_correct_answer','response_type']].groupby(['model','response_type']).mean().reset_index(),
        ax = ax,
        label = 'Normal Model',
    )

    sns.scatterplot(
        x = 'model',
        y ='is_correct_answer',
        data = df_post[df_post['response_type'] == 'order_independent'][['model','is_correct_answer','response_type']].groupby(['model','response_type']).mean().reset_index(),
        ax = ax,
        s = 100,
        label = 'Post-Finetuning Order Independent Model',
        zorder = 5      # Force it to be on top
    )
    
    sns.scatterplot(
        x = 'model',
        y ='is_correct_answer',
        data = df[df['response_type'] == 'order_independent'][['model','is_correct_answer','response_type']].groupby(['model','response_type']).mean().reset_index(),
        ax = ax,
        s = 100,
        label = 'Pre-Finetuning Order Independent Model',
        zorder = 4      # Force it to be near the top
    )
    
    #ax.set_ylim([.15,.4])
    ax.set_ylim([0.1,0.6])
    ax.set_xticklabels([l._text.split('/')[-1] for l in ax.get_xticklabels()], rotation=90, ha='right')
    
    ax.legend(ax.get_legend_handles_labels()[0][-2:],ax.get_legend_handles_labels()[1][-2:],bbox_to_anchor=(1,1),loc = 'upper left')
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{dataset_type} Top 1 Accuracy")

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

def create_perplexity_plot(perplexity_data):
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
    
    ax.set_title(f"WikiText-103 Perplexity - Finetuning")
    ax.set_xlabel("")
    ax.set_ylabel("Perplexity")
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"plots/finetuning_perplexity.png", bbox_inches='tight')
    plt.close()

def analyze_dataset(dataset_type):
    """Analyze results for either MMLU or CSQA"""
    model_results = []
    
    # Collect all model results first
    for pair in model_pairs:
        # Load pre-finetuning results
        pre_results = load_results(pair[f"pre_path_{dataset_type.lower()}"])
        if pre_results is None:
            continue
            
        # Load post-finetuning results
        post_results = load_results(pair[f"post_path_{dataset_type.lower()}"])

        pre_results_perm = load_results(pair[f"pre_path_{dataset_type.lower()}_perm"])
        
        model_results.append({
            'name': pair['name'],
            'pre_data': pre_results,
            'post_data': post_results,
            'pre_data_perm': pre_results_perm,
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


def analyze_perplexity(model_pairs):
    # Create perplexity plot (unchanged)
    perplexity_data = {}
    for pair in model_pairs:
        perp_results = load_perplexity_results(pair["output_dir"])
        if perp_results is not None:
            perplexity_data[pair["name"]] = perp_results
    
    create_perplexity_plot(
        perplexity_data,
    )

def main():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Analyze both datasets
    analyze_dataset("MMLU")
    analyze_dataset("CSQA")
    analyze_perplexity(model_pairs)

if __name__ == "__main__":
    main() 