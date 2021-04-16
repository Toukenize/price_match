import torch.nn as nn
from transformers import AutoModel


class ShopeeNLPModel(nn.Module):

    def __init__(self, model_path, num_classes, dropout_prob=0.2):

        super(ShopeeNLPModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(self.model.config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lin = nn.Linear(self.model.config.hidden_size, num_classes)
        self.feature_dim = self.model.config.hidden_size

    def forward(self, **data):

        x = self.model(**data).last_hidden_state[:, 0, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.lin(x)
        x = self.tanh(x)

        return x

    def extract_features(self, **data):

        x = self.model(**data).last_hidden_state[:, 0, :]
        x = self.norm(x)

        return x
