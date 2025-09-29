from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import pandas as pd
from models.dataset_2_classifier import FusionModel
import os


def train_and_evaluate_mlp(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    image_df: pd.DataFrame,
    checkpoint_dir: str,
    mode: str,
    inference_only: bool,
):
    """
    Train / evaluate a small MLP (optionally fused with tabular features) using GroupKFold CV.
    - embeddings: array-like of image embeddings (num_samples x embedding_dim)
    - labels: binary labels (0/1)
    - groups: group ids for GroupKFold (e.g., patient ids)
    - image_df: pandas DataFrame with tabular features (Age, Gender, etc.)
    - mode: "unimodal" | "multimodal_best_features" | "multimodal_all_features"
    - inference_only: if True, load checkpoints for each fold; else train
    Returns: mean metrics dict and per-fold lists dict
    """

    # sets deterministic behavior for reproduction
    seed = 37
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device for tensors / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inference_only:
        print(
            f"=========================Selected mode: {mode} - Inference========================"
        )
    else:
        print(
            f"=========================Selected mode: {mode} - Training + Inference========================"
        )

    # outer CV that respects groups
    gkf = GroupKFold(n_splits=5)

    # normalizes Age with z-score; keeps binary features as-is (0/1)
    image_df_norm = image_df.copy()
    image_df_norm["Age"] = (image_df["Age"] - image_df["Age"].mean()) / image_df[
        "Age"
    ].std()

    # default features to use
    feature_cols = ["Age", "Gender", "Alcohol", "Smoking", "Chewing_Betel_Quid "]

    # picks subset depending on requested mode
    if mode == "multimodal_all_features":
        feature_cols = ["Age", "Gender", "Alcohol", "Smoking", "Chewing_Betel_Quid "]
    else:
        # best features
        feature_cols = ["Gender", "Chewing_Betel_Quid "]

    # encoding for downstream tasks
    features_all = image_df_norm[feature_cols].values.astype(np.float32)

    # collects metrics across folds
    all_fold_accuracy = []
    all_fold_f1 = []
    all_fold_precision = []
    all_fold_recall = []

    # keeps track of best fold
    best_acc = -1
    best_fold_idx = None
    best_test_indices = None
    best_test_labels = None
    best_test_preds = None

    # iterates over folds using GroupKFold
    for fold, (trainval_idx, test_idx) in enumerate(
        gkf.split(embeddings, labels, groups), 1
    ):
        print(f"\n========== Fold {fold} ==========")

        embeddings = np.array(embeddings)
        labels = np.array(labels)
        groups = np.array(groups)

        # creates a train/val split inside trainval using groups
        gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=fold)
        train_sub_idx, val_sub_idx = next(
            gss.split(
                embeddings[trainval_idx], labels[trainval_idx], groups[trainval_idx]
            )
        )
        train_idx = trainval_idx[train_sub_idx]
        val_idx = trainval_idx[val_sub_idx]

        train_embeddings, val_embeddings, test_embeddings = (
            embeddings[train_idx],
            embeddings[val_idx],
            embeddings[test_idx],
        )
        train_labels, val_labels, test_labels = (
            labels[train_idx],
            labels[val_idx],
            labels[test_idx],
        )

        train_features, val_features, test_features = (
            features_all[train_idx],
            features_all[val_idx],
            features_all[test_idx],
        )

        # builds TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float32),
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_embeddings, dtype=torch.float32),
            torch.tensor(val_features, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.long),
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_embeddings, dtype=torch.float32),
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long),
        )

        # small loaders shuffles only training data
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # builds fusion model according to chosen mode
        embedding_dim = train_embeddings.shape[1]
        if mode == "unimodal":
            model = FusionModel(embedding_dim=embedding_dim, feature_dim=0).to(device)
        elif mode == "multimodal_best_features":
            model = FusionModel(embedding_dim=embedding_dim, feature_dim=2).to(device)
        elif mode == "multimodal_all_features":
            model = FusionModel(embedding_dim=embedding_dim, feature_dim=5).to(device)

        # handles class imbalance by scaling pos_weight for BCEWithLogitsLoss
        pos_weight_value = (
            (0.98 * np.sum(train_labels == 0) / np.sum(train_labels == 1))
            if np.sum(train_labels == 1) > 0
            else 1.0
        )
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # training/testing
        if inference_only:
            ckpt_path = os.path.join(
                checkpoint_dir, f"fold_{fold}_epoch_280_{mode}_checkpoint.pt"
            )
            print(f"\n[Fold {fold}] Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # training
            num_epochs = 280
            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(num_epochs):
                # training step
                model.train()
                train_loss = 0.0
                for img_emb, feats, targets in train_loader:
                    img_emb, feats, targets = (
                        img_emb.to(device),
                        feats.to(device),
                        targets.to(device).float().unsqueeze(1),
                    )
                    optimizer.zero_grad()
                    outputs = model(img_emb, feats)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * img_emb.size(0)
                train_loss /= len(train_loader.dataset)

                # validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for img_emb, feats, targets in val_loader:
                        img_emb, feats, targets = (
                            img_emb.to(device),
                            feats.to(device),
                            targets.to(device).float().unsqueeze(1),
                        )
                        outputs = model(img_emb, feats)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * img_emb.size(0)
                val_loss /= len(val_loader.dataset)

                # test loss monitoring
                test_loss = 0.0
                with torch.no_grad():
                    for img_emb, feats, targets in test_loader:
                        img_emb, feats, targets = (
                            img_emb.to(device),
                            feats.to(device),
                            targets.to(device).float().unsqueeze(1),
                        )
                        outputs = model(img_emb, feats)
                        loss = criterion(outputs, targets)
                        test_loss += loss.item() * img_emb.size(0)
                test_loss /= len(test_loader.dataset)

                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}"
                )

                # last epoch model saving
                if epoch == num_epochs - 1:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

                    # save checkpoint for this fold (local file)
                    ckpt_path = f"fold_{fold}_epoch_{epoch + 1}_unimodal_checkpoint.pt"
                    torch.save(
                        {
                            "fold": fold,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "test_loss": test_loss,
                            "train_loss": train_loss,
                        },
                        ckpt_path,
                    )
                    print(f"Saved checkpoint for fold {fold} at {ckpt_path}")

            model.load_state_dict(best_model_state)

        # testing
        model.eval()
        all_preds, all_targets = [], []

        if mode == "unimodal":
            margin = 0.45
        else:
            margin = 0.5

        with torch.no_grad():
            for img_emb, feats, targets in test_loader:
                img_emb, feats, targets = (
                    img_emb.to(device),
                    feats.to(device),
                    targets.to(device).float().unsqueeze(1),
                )
                outputs = model(img_emb, feats)
                preds = (
                    torch.sigmoid(outputs) > margin
                ).float()  # converting logits -> prob -> binary
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # metrics calculation
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro")
        prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)

        all_fold_accuracy.append(acc)
        all_fold_f1.append(f1)
        all_fold_precision.append(prec)
        all_fold_recall.append(rec)

        if acc > best_acc:
            best_acc = acc
            best_fold_idx = fold
            best_test_indices = test_idx
            best_test_labels = np.array(all_targets)
            best_test_preds = np.array(all_preds)

        print(
            f"Fold {fold} Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}"
        )

        # plots and displays confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6), dpi=150)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Benign", "OPMD + OCA"]
        )
        disp.plot(cmap="PuBuGn", text_kw={"fontsize": 12})
        plt.title(f"Confusion Matrix Fold {fold}", fontsize=14)
        plt.tight_layout()
        plt.show()

    # prints cross-validation summary (mean ± std)
    print("\n========== Cross-Validation Results ==========")
    print(
        f"Mean Accuracy: {np.mean(all_fold_accuracy):.4f} ± {np.std(all_fold_accuracy):.4f}"
    )
    print(f"Mean Macro F1: {np.mean(all_fold_f1):.4f} ± {np.std(all_fold_f1):.4f}")
    print(
        f"Mean Precision: {np.mean(all_fold_precision):.4f} ± {np.std(all_fold_precision):.4f}"
    )
    print(
        f"Mean Recall: {np.mean(all_fold_recall):.4f} ± {np.std(all_fold_recall):.4f}"
    )

    # saves best fold info to disk for later inspection if any fold was found
    if best_test_indices is not None:
        np.save("best_fold_test_indices.npy", best_test_indices)
        np.save("best_fold_test_labels.npy", best_test_labels)
        np.save("best_fold_test_preds.npy", best_test_preds)
        print("Saved best fold test indices, labels, and predictions as .npy files.")

    return {
        "accuracy": np.mean(all_fold_accuracy),
        "f1": np.mean(all_fold_f1),
        "precision": np.mean(all_fold_precision),
        "recall": np.mean(all_fold_recall),
    }, {
        "accuracy": all_fold_accuracy,
        "f1": all_fold_f1,
        "precision": all_fold_precision,
        "recall": all_fold_recall,
    }
