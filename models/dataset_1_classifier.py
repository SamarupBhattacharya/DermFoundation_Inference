import torch
import torch.nn as nn


# Simple MLP used for dataset 1 experiments.
# - input_dim: size of input embedding/features
# - hidden_dim: width of hidden layers
# - num_layers: total number of linear layers (including first mapping from input)
# Produces a single logit per example (use with BCEWithLogitsLoss).
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=10):
        super(MLPClassifier, self).__init__()
        layers = []
        # first layer: map input -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())  # non-linearity after first linear

        # stack (num_layers - 1) hidden layers, each hidden_dim -> hidden_dim
        # keeps architecture simple and deep if num_layers large
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # final linear head -> single logit for binary classification
        layers.append(nn.Linear(hidden_dim, 1))  # output shape: (batch_size, 1) logits

        # pack everything into a Sequential for tidy forward pass
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: (batch_size, input_dim)
        # returns logits shape: (batch_size, 1)
        return self.model(x)
