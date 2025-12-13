import argparse

from util.dataset_analysis import run_dataset_exploration
from inference import run_inference

import torch

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
        choices=["analysis", "inference"],
        help="Run dataset analysis or model inference"
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

    args = parser.parse_args()

    if args.mode == "analysis":
        run_dataset_exploration(dataset_name=args.dataset,
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


if __name__ == "__main__":
    main()
