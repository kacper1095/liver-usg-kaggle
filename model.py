import torch.nn as nn
import torch.nn.functional as F

from custom_densenet import densenet121


class PretrainedModel(nn.Module):
    def __init__(self, extract_embeddings: bool = False):
        super().__init__()
        pretrained_model = densenet121(pretrained=True, drop_rate=0.1).train()
        self.extractor = pretrained_model.features
        self.classifier = nn.Linear(pretrained_model.classifier.in_features, 2)

        self.extract_embeddings = extract_embeddings

    def forward(self, x):
        is_test = len(x.size()) == 5
        if is_test:
            b, ncrops, c, h, w = x.size()
            features = self.extractor(x.view(-1, c, h, w))
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(
                features.size(0), -1)
            out = F.dropout(pooled_features, p=0.1, training=self.training)
            out = self.classifier(out).view(b, ncrops, -1).mean(dim=1)
        else:
            features = self.extractor(x)
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(
                features.size(0), -1)
            out = F.dropout(pooled_features, p=0.1, training=self.training)
            out = self.classifier(out)

        if self.extract_embeddings:
            if is_test:
                pooled_features = pooled_features.view(b, ncrops, -1).mean(dim=1)
            return out, pooled_features

        return out
