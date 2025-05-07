import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    """Base class for backbone architecture."""
    def __init__(self, backbone_type='vgg16', pretrained=True):
        super(Backbone, self).__init__()
        if backbone_type == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained).features
        elif backbone_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Remove the fully connected layers from ResNet
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        else:
            raise ValueError(f"Backbone {backbone_type} not supported")

    def forward(self, x):
        return self.model(x)
