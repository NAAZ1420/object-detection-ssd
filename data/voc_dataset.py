import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import transforms  # If you're using the transforms module from above

class VOCDataset(Dataset):
    """Custom dataset class for VOC."""
    
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "Annotations")
        self.image_ids = list(sorted(os.listdir(self.image_dir)))
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annotation_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.xml'))
        
        image = Image.open(img_path).convert("RGB")
        target = self.parse_annotation(annotation_path)
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def parse_annotation(self, annotation_path):
        """Parse the XML annotations and extract the bounding boxes."""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.iter("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)  # You might want to map these to integer labels
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels)
        }
        return target
