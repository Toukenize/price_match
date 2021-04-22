import torch.nn as nn
from transformers import AutoModel


class ShopeeNLPModel(nn.Module):

    def __init__(self, model_path, num_classes, dropout,
                 margin_func, **margin_params):

        super(ShopeeNLPModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = self.model.config.hidden_size
        self.arc_margin = margin_func(
            in_features=self.feature_dim,
            out_features=num_classes,
            **margin_params)

    def forward(self, input_ids, token_type_ids, attention_mask, label):

        x = self.extract_features(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        x = self.dropout(x)
        x = self.arc_margin(x, label)

        return x

    def extract_features(self, **data):

        x = self.model(**data).last_hidden_state[:, 0, :]

        return x
