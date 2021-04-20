import torch
import torch.nn as nn


class ShopeeIMGModel(nn.Module):

    def __init__(self, model_path, num_classes, margin_func, **margin_params):
        super(ShopeeIMGModel, self).__init__()

        self.model = torch.load(model_path)
        self.model.classifier = nn.Identity()
        self.feature_dim = self.model.num_features
        self.arc_margin = margin_func(
            in_features=self.feature_dim,
            out_features=num_classes,
            **margin_params)

    def forward(self, image, label):

        x = self.extract_features(image)
        x = self.arc_margin(x, label)

        return x

    def extract_features(self, image):

        x = self.model(image)

        return x
