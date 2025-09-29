import os
import sys
import argparse
import numpy as np
import pandas as pd
from utils.core_utils import train_and_evaluate_mlp


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate MLP for OSCC project."
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing or saving checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing the required .npy and .csv data files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unimodal", "multimodal_best_features", "multimodal_all_features"],
        default="multimodal_all_features",
        help="Training mode to use.",
    )
    parser.add_argument(
        "--inference_only",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether to run in inference-only mode (default: yes).",
    )

    args = parser.parse_args()

    # ---- Print parsed arguments for clarity ----
    print("\n===== OSCC Project: Main Script =====")
    print(f"Checkpoint Directory : {args.checkpoint_dir}")
    print(f"Data Directory        : {args.data_dir}")
    print(f"Mode                  : {args.mode}")
    print(f"Inference Only        : {args.inference_only}")
    print("=====================================\n")

    # ---- Validate directories ----
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory '{args.checkpoint_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        sys.exit(1)

    # ---- Validate required data files ----
    required_files = [
        "Imagewise_Data.csv",
        "Patientwise_Data.csv",
        "dataset_2_embeddings.npy",
        "dataset_2_labels.npy",
    ]
    for f in required_files:
        if not os.path.exists(os.path.join(args.data_dir, f)):
            print(f"Error: Required file '{f}' not found in '{args.data_dir}'.")
            sys.exit(1)

    # ---- Load data ----
    print("Loading data...")
    image_df = pd.read_csv(os.path.join(args.data_dir, "Imagewise_Data.csv"))
    patient_df = pd.read_csv(os.path.join(args.data_dir, "Patientwise_Data.csv"))

    # --- Preprocess ---
    print("Preprocessing data...")
    image_df = image_df[image_df["Category"] != "Healthy"]
    image_df["Patient ID"] = image_df["Image Name"].str.rsplit("-", n=1).str[0]
    image_df = pd.merge(image_df, patient_df, on="Patient ID", how="left")
    groups = image_df["Patient ID"].tolist()

    image_df = image_df.drop(["Image Name", "Patient ID"], axis=1)
    image_df["Age"] = image_df["Age"].fillna(image_df["Age"].mean())
    image_df["Gender"] = image_df["Gender"].fillna("M").map({"M": 1, "F": 0})
    image_df["Image Count"] = image_df["Image Count"].fillna(0)

    # Encode Clinical Diagnosis based on custom scoring
    diagnoses = {
        diag: 10
        * image_df[
            (image_df["Category"] == "OCA") & (image_df["Clinical Diagnosis"] == diag)
        ].shape[0]
        + image_df[
            (image_df["Category"] == "OPMD") & (image_df["Clinical Diagnosis"] == diag)
        ].shape[0]
        - image_df[
            (image_df["Category"] == "Benign")
            & (image_df["Clinical Diagnosis"] == diag)
        ].shape[0]
        for diag in image_df["Clinical Diagnosis"].unique()
    }
    sorted_diags = sorted(diagnoses.items(), key=lambda x: x[1])
    encoding_dict = {diag: idx for idx, (diag, _) in enumerate(sorted_diags)}
    image_df["Clinical Diagnosis"] = image_df["Clinical Diagnosis"].map(encoding_dict)

    # Encode habits
    for col in ["Smoking", "Chewing_Betel_Quid ", "Alcohol"]:
        image_df[col] = image_df[col].fillna("No").map({"Yes": 1, "No": 0})

    # Encode category (target)
    image_df["Category"] = image_df["Category"].map({"Benign": 0, "OPMD": 1, "OCA": 1})

    # ---- Load embeddings and labels ----
    print("Loading embeddings and labels...")
    embeddings = np.load(os.path.join(args.data_dir, "dataset_2_embeddings.npy"))[:-748]
    labels = np.load(os.path.join(args.data_dir, "dataset_2_labels.npy"))[:-748]

    print("Data loaded successfully.")
    print(f"Missing values remaining: {image_df.isna().sum().sum()}")
    print(
        f"Number of numeric columns: {len(image_df.select_dtypes(include=[np.number]).columns)}"
    )

    # ---- Convert inference_only string to bool ----
    inference_only = args.inference_only.lower() == "yes"

    # ---- Call training/evaluation ----
    print("\nStarting MLP process...\n")
    train_and_evaluate_mlp(
        embeddings=embeddings,
        labels=labels,
        groups=groups,
        image_df=image_df,
        checkpoint_dir=args.checkpoint_dir,
        mode=args.mode,
        inference_only=inference_only,
    )


if __name__ == "__main__":
    main()
