import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

class RandomHorizontalFlip(object):
    """Randomly flip the image and its target horizontally."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = self.flip_target(target)
        return image, target

    def flip_target(self, target):
        # Flip bounding box coordinates (if applicable)
        target = target.copy()
        target['boxes'] = target['boxes'][:, [2, 3, 0, 1]]  # Flip x and y coordinates
        return target

class ToTensor(object):
    """Convert a PIL Image or numpy.ndarray to tensor."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    """Normalize the image using mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# Example of using the transforms
transform = transforms.Compose([
    RandomHorizontalFlip(prob=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
