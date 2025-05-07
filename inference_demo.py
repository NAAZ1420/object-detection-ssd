import torch
from model import SSDHead
import cv2

# Load model
model = SSDHead(2048, num_classes=21)
model.load_state_dict(torch.load("weights/best_model.pth"))
model.eval()

# Run inference on a sample image
image = cv2.imread('sample_image.jpg')
locs, confs = model(image)
# Visualize the results by drawing bounding boxes on image
