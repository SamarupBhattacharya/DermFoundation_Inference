import os
import sys
import argparse
import numpy as np
from utils.core_utils_1 import train_and_evaluate_mlp


def main():
    # build CLI so reviewer can run script from terminal with paths
    parser = argparse.ArgumentParser(
        description="Train or evaluate MLP for OSCC project for Dataset 1(Kaggle Dataset)."
    )

    # where checkpoints live (or will be saved by training)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing or saving checkpoints.",
    )
    # folder with precomputed .npy embeddings/labels
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing the required .npy files.",
    )

    # run inference-only or run training+inference; kept as string for simple CLI
    parser.add_argument(
        "--inference_only",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether to run in inference-only mode (default: yes).",
    )

    args = parser.parse_args()

    # print parsed args so logs show what was used
    print("\n===== OSCC Project(Kaggle Dataset): Main Script =====")
    print(f"Checkpoint Directory  : {args.checkpoint_dir}")
    print(f"Data Directory        : {args.data_dir}")
    print(f"Inference Only        : {args.inference_only}")
    print("=====================================\n")

    # basic sanity checks: fail fast if directories missing
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory '{args.checkpoint_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        sys.exit(1)

    # expect these exact filenames in data_dir (precomputed)
    embeddings = np.load(os.path.join(args.data_dir, "dataset_1_embeddings.npy"))
    labels = np.load(os.path.join(args.data_dir, "dataset_1_labels.npy"))

    # call the training/eval helper. convert CLI "yes"/"no" to bool
    train_and_evaluate_mlp(
        embeddings,
        labels,
        checkpoint_dir=args.checkpoint_dir,
        inference_only=args.inference_only == "yes",
    )


if __name__ == "__main__":
    # entrypoint when run as a script
    main()
