import torch
import torch.nn as nn


class ShopeeIMGModel(nn.Module):

    def __init__(self, model_path, num_classes, dropout_prob=0.2):
        super(ShopeeIMGModel, self).__init__()

        self.model = torch.load(model_path)
        self.model.classifier = nn.Identity()
        self.norm = nn.LayerNorm(self.model.num_features)
        self.dropout = nn.Dropout(dropout_prob)
        self.lin = nn.Linear(self.model.num_features, num_classes)
        self.tanh = nn.Tanh()
        self.feature_dim = self.model.num_features

    def forward(self, image):

        x = self.model(image)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.lin(x)
        x = self.tanh(x)

        return x

    def extract_features(self, image):

        x = self.model(image)
        x = self.norm(x)

        return x
