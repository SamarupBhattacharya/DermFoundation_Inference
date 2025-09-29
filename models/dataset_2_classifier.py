import torch
import torch.nn as nn


# simple fully-connected block to embed tabular features into same space as image embedding
# feature_dim: number of input tabular features (e.g., 2 or 5)
# returns tensor of shape (batch_size, 256)
class FeatureFC(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureFC, self).__init__()
        # tiny MLP: map raw features -> 256-d vector, apply non-linearity
        self.model = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU())

    def forward(self, x):
        # x expected shape: (batch_size, feature_dim), dtype float
        return self.model(x)


# Fusion wrapper that optionally concatenates image embedding and feature embedding,
# then forwards through a classifier MLP.
# embedding_dim: dimension of image embedding input
# feature_dim: if 0 => unimodal (only image embeddings), otherwise uses FeatureFC to map tabular features
class FusionModel(nn.Module):
    def __init__(
        self, embedding_dim, feature_dim=2, fused_dim=256, hidden_dim=8, num_layers=2
    ):
        super(FusionModel, self).__init__()
        # projector for tabular features, if feature_dim==0 this still exists but won't be used
        self.feature_fc = FeatureFC(feature_dim)
        # if multimodal training/testing, classifier input dim = image_emb + projected features
        if feature_dim > 0:
            self.mlp = MLPClassifier(
                input_dim=embedding_dim + fused_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            # unimodal: classifier takes only image embedding
            self.mlp = MLPClassifier(
                input_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers
            )
        self.feature_dim = feature_dim

    def forward(self, image_embed, features):
        # image_embed: (batch_size, embedding_dim)
        # features: (batch_size, feature_dim)
        if self.feature_dim > 0:
            # map features -> fused_dim (256), then concats with image embedding
            feat_embed = self.feature_fc(features)  # (batch_size, fused_dim)
            fused = torch.cat(
                [image_embed, feat_embed], dim=1
            )  # concats along feature dimension => (batch_size, embedding_dim + fused_dim)
        else:
            # no tabular features, passes image embedding directly
            fused = image_embed
        # classifier returns single logit per sample
        return self.mlp(fused)


# MLP classifier producing a single logit (binary classification).
# input_dim: dimensionality of input features to classifier
# hidden_dim: final hidden layer dim before output
# num_layers: number of intermediate linear layers (upper-bounded by len(prev_dim)+1)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=10):
        super(MLPClassifier, self).__init__()
        layers = []

        prev_dim = [16, 8, 8, 8, 64, 256, 256, 256, 256]
        # first map input_dim -> prev_dim[0]
        layers.append(nn.Linear(input_dim, prev_dim[0]))
        no = 1

        layers.append(nn.LeakyReLU())
        for i in range(num_layers - 1):
            if i != num_layers - 2:
                layers.append(nn.Linear(prev_dim[i], prev_dim[i + 1]))
                # layers.append(nn.BatchNorm1d(prev_dim[i+1]))
            else:
                layers.append(nn.Linear(prev_dim[i], hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.4 / no))
            no += 1

        layers.append(nn.Linear(hidden_dim, 1))  # Binary output

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
