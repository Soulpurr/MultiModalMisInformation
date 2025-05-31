import torch.nn as nn
import timm

class XceptionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionModel, self).__init__()
        self.model = timm.create_model('xception', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
