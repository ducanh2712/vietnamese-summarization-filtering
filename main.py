import argparse
import torch
import yaml
from pathlib import Path

from util.dataset_analysis import run_dataset_exploration
from inference import run_inference
from train import train_model
from eval import evaluate_model

# NEW IMPORTS
from data_processing.data_processor import process_dataset_all_approaches

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_data_processing(config):
    """
    Run data processing cho các approaches
    
    Args:
        config: Configuration dict
    """
    print("=" * 80)
    print("STARTING DATA PROCESSING")
    print("=" * 80)
    
    # Extract config
    dataset_name = config['dataset']['name']
    processing_config = config.get('data_processing', {})
    
    approaches = processing_config.get('approaches', ['baseline'])
    max_tokens = processing_config.get('max_tokens', 512)
    output_dir = processing_config.get('output_dir', './data/processed')
    limit = processing_config.get('limit_samples', None)
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Approaches: {', '.join(approaches)}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Output dir: {output_dir}")
    if limit:
        print(f"   Sample limit: {limit}")
    
    # Process
    results = process_dataset_all_approaches(
        dataset_name=dataset_name,
        approaches=approaches,
        max_tokens=max_tokens,
        output_dir=output_dir,
        limit=limit
    )
    
    print("\n" + "=" * 80)
    print("DATA PROCESSING COMPLETED")
    print("=" * 80)
    print(f"\n📊 Results:")
    for approach, stats in results.items():
        print(f"   {approach}: {stats['num_pairs']} pairs processed")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Summarization Project"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analysis", "inference", "train", "eval", "process_data"],  # NEW
        help="Run mode: analysis, inference, train, eval, or process_data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="8Opt/vietnamese-summarization-dataset-0001",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index for inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model for evaluation or inference"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split for evaluation"
    )
    
    # NEW: Data processing specific arguments
    parser.add_argument(
        "--approaches",
        type=str,
        nargs='+',
        default=None,
        choices=['baseline', 'fused_sentences', 'discourse_aware', 'all'],
        help="Which preprocessing approaches to run (for process_data mode)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # GPU info
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current GPU device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # Route to appropriate function
    if args.mode == "analysis":
        run_dataset_exploration(
            dataset_name=args.dataset,
            split="train",
            n_samples=3,
            n_vocab=15,
            export_n=100
        )
    
    elif args.mode == "process_data":
        # Override config với command line args nếu có
        if args.approaches:
            config['data_processing']['approaches'] = args.approaches
        if args.limit:
            config['data_processing']['limit_samples'] = args.limit
        
        run_data_processing(config)
    
    elif args.mode == "inference":
        print("=" * 80)
        print("STARTING INFERENCE")
        print("=" * 80)
        run_inference(
            sample_index=args.sample_index,
            dataset_name=args.dataset
        )
        print("\n" + "=" * 80)
        print("INFERENCE COMPLETED")
        print("=" * 80)
    
    elif args.mode == "train":
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        trainer, model_path = train_model(config_path=args.config)
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED - STARTING EVALUATION")
        print("=" * 80)
        evaluate_model(
            model_path=model_path, 
            config_path=args.config, 
            split=args.split
        )
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED")
        print("=" * 80)
    
    elif args.mode == "eval":
        print("=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)
        evaluate_model(
            model_path=args.model_path,
            config_path=args.config,
            split=args.split
        )
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED")
        print("=" * 80)


if __name__ == "__main__":
    main()