import argparse
import torch

from util.dataset_analysis import run_dataset_exploration
from inference import run_inference
from train import train_model
from eval import evaluate_model

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Summarization Project"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analysis", "inference", "train", "eval"],
        help="Run dataset analysis, model inference, training, or evaluation"
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

    args = parser.parse_args()

    if args.mode == "analysis":
        run_dataset_exploration(
            dataset_name=args.dataset,
            split="train",
            n_samples=3,
            n_vocab=15,
            export_n=100
        )

    elif args.mode == "inference":
        run_inference(
            sample_index=args.sample_index,
            dataset_name=args.dataset
        )

    elif args.mode == "train":
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        trainer, model_path = train_model(config_path=args.config)
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED - STARTING EVALUATION")
        print("=" * 80)
        evaluate_model(model_path=model_path, config_path=args.config, split=args.split)

    elif args.mode == "eval":
        print("=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)
        evaluate_model(
            model_path=args.model_path,
            config_path=args.config,
            split=args.split
        )


if __name__ == "__main__":
    main()