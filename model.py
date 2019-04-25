import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_model = models.densenet121(pretrained=True, drop_rate=0.2)
        self.extractor = pretrained_model.features
        self.classifier = nn.Linear(pretrained_model.classifier.in_features, 2)

    def forward(self, x):
        features = self.extractor(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out
