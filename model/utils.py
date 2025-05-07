import torch
import numpy as np

def intersection_over_union(box1, box2):
    """Calculate the Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def non_maximum_suppression(predictions, iou_threshold=0.5):
    """Perform Non-Maximum Suppression (NMS) on predictions."""
    boxes, scores = predictions['boxes'], predictions['scores']
    
    # Sort the boxes by score in descending order
    idxs = torch.argsort(scores, descending=True)
    keep = []
    
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        
        if idxs.numel() == 1:
            break
        
        other_boxes = boxes[idxs[1:]]
        iou = torch.tensor([intersection_over_union(boxes[i], other_box) for other_box in other_boxes])
        
        # Remove boxes with IoU greater than the threshold
        idxs = idxs[1:][iou <= iou_threshold]
    
    return boxes[keep], scores[keep]
