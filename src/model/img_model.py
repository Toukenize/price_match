import torch
import torch.nn as nn


class ShopeeIMGModel(nn.Module):

    def __init__(self, model_path, num_classes, dropout,
                 margin_func, feature_dim=512, **margin_params):
        super(ShopeeIMGModel, self).__init__()

        self.feature_dim = feature_dim

        self.model = torch.load(model_path)

        if 'efficientnet' in model_path:
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()

        elif 'nfnet' in model_path:
            self.model.head.fc = nn.Identity()
            self.model.head.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(self.model.num_features, self.feature_dim)
        self.bn = nn.BatchNorm1d(self.feature_dim)
        self.arc_margin = margin_func(
            in_features=self.feature_dim,
            out_features=num_classes,
            **margin_params)

    def forward(self, image, label):

        x = self.extract_features(image)
        x = self.arc_margin(x, label)

        return x

    def extract_features(self, image):

        batch_size = image.shape[0]
        x = self.model(image)
        x = self.pooling(x).view(batch_size, -1)
        x = self.dropout(x)
        x = self.lin(x)
        x = self.bn(x)

        return x
