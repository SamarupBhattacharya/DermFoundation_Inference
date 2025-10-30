import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
import random
import os
from models.dataset_1_classifier import MLPClassifier


# tiny dataset wrapper around embeddings + labels so DataLoader works
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        # store tensors on CPU; moves to device done in training loop
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # returns (embedding, label) for idx
        return self.embeddings[idx], self.labels[idx]


# helper to fix randomness so runs are reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make CUDA deterministic (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate_mlp(
    embeddings: np.ndarray,
    labels: np.ndarray,
    checkpoint_dir: str = None,
    inference_only: bool = False,
):
    """
    Train/eval simple MLP over embedding vectors using Stratified K-Fold.
    - embeddings: (N, D) numpy array
    - labels: (N,) binary labels (0/1)
    - checkpoint_dir: folder where checkpoints live (used when inference_only)
    - inference_only: if True, skip training and load checkpoint per fold
    """

    n_splits = 5
    set_seed(42)  # reproducible splits and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ensure numpy arrays for sklearn indexing
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # stratified to keep class balance in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # store per-fold metrics
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    fold = 1
    for train_idx, test_idx in skf.split(embeddings, labels):
        # per-fold run
        print(f"\n=== Fold {fold} ===")
        set_seed(42)  # reset seed per fold for consistent initialization

        # split arrays
        train_embeddings, test_embeddings = embeddings[train_idx], embeddings[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        # wrap into Dataset objects for DataLoader
        train_dataset = EmbeddingDataset(train_embeddings, train_labels)
        test_dataset = EmbeddingDataset(test_embeddings, test_labels)
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # build model sized to embedding dimension
        embedding_dim = train_embeddings.shape[1]
        model = MLPClassifier(input_dim=embedding_dim, hidden_dim=128, num_layers=5).to(
            device
        )

        # BCEWithLogits expects float targets; optimizer + scheduler typical
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.2
        )

        if inference_only:
            # load saved state dict (assumes whole dict saved earlier)
            ckpt_path = os.path.join(
                checkpoint_dir, f"dataset_1_mlp_fold_{fold}_last_epoch.pth"
            )
            print(f"\n[Fold {fold}] Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
        else:
            # training loop
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                # tqdm for progress, iterate batches
                for inputs, targets in tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
                ):
                    # move batch to device and convert labels to float column vector
                    inputs, targets = (
                        inputs.to(device),
                        targets.to(device).float().unsqueeze(1),
                    )
                    optimizer.zero_grad()
                    outputs = model(inputs)  # logits shape (B,1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss /= len(train_loader.dataset)

                # evaluate on test set as validation proxy (no grads)
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = (
                            inputs.to(device),
                            targets.to(device).float().unsqueeze(1),
                        )
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(test_loader.dataset)
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                # scheduler reduces LR when val loss plateaus
                scheduler.step(val_loss)

        # final evaluation on test set (either after training or after loading checkpoint)
        model.eval()
        true_labels, predicted_labels = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = (
                    inputs.to(device),
                    targets.to(device).float().unsqueeze(1),
                )
                outputs = model(inputs)  # logits
                preds = (torch.sigmoid(outputs) > 0.5).float()  # prob -> binary
                true_labels.extend(targets.cpu().numpy().flatten())
                predicted_labels.extend(preds.cpu().numpy().flatten())

        # compute metrics (macro to treat classes equally)
        acc = accuracy_score(true_labels, predicted_labels)
        prec = precision_score(
            true_labels, predicted_labels, average="macro", zero_division=0
        )
        rec = recall_score(
            true_labels, predicted_labels, average="macro", zero_division=0
        )
        f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        print(
            f"Accuracy: {acc:.4f}, Macro Precision: {prec:.4f}, Macro Recall: {rec:.4f}, Macro F1: {f1:.4f}"
        )

        # save per-fold metrics
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        # show confusion matrix to eyeball errors per fold
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6), dpi=300)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Non-Cancerous", "Cancerous"]
        )
        disp.plot(cmap="PuBuGn", text_kw={"fontsize": 12})
        plt.title(f"Fold {fold} Confusion Matrix", fontsize=16)
        plt.show()

        fold += 1

    # print averaged metrics across folds for quick summary
    print("\n=== Average metrics across all folds ===")
    print(f"Accuracy: {np.mean(acc_list):.4f}")
    print(f"Macro Precision: {np.mean(prec_list):.4f}")
    print(f"Macro Recall: {np.mean(rec_list):.4f}")
    print(f"Macro F1: {np.mean(f1_list):.4f}")
