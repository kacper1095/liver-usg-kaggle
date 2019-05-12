import torch.nn as nn
import torch.nn.functional as F

from custom_densenet import densenet121


class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_model = densenet121(pretrained=False, drop_rate=0.1).train()
        self.extractor = pretrained_model.features
        self.classifier = nn.Linear(pretrained_model.classifier.in_features, 2)

    def forward(self, x):
        features = self.extractor(x)
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.classifier(out)
        return out
