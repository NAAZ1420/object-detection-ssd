import torch
import torch.nn as nn
import torch.nn.functional as F

class SSDHead(nn.Module):
    """SSD Head: generates detections from feature maps."""
    def __init__(self, num_classes, input_channels=512, num_priors=8732):
        super(SSDHead, self).__init__()
        
        # Classification and Localization Prediction layers
        self.classification = nn.Conv2d(input_channels, num_priors * num_classes, kernel_size=3, padding=1)
        self.localization = nn.Conv2d(input_channels, num_priors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        loc_preds = self.localization(x)
        class_preds = self.classification(x)
        
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        class_preds = class_preds.permute(0, 2, 3, 1).contiguous()
        
        loc_preds = loc_preds.view(loc_preds.size(0), -1, 4)
        class_preds = class_preds.view(class_preds.size(0), -1, num_classes)
        
        return loc_preds, class_preds
