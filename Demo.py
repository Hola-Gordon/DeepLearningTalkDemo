import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import time
import random


# Set up colors for visualization (vibrant colors for better visual impact)
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), 
          (255, 0, 255), (80, 70, 180), (250, 80, 190), (245, 145, 50)]

# COCO dataset classes (only keep the ones we want to detect)
# You can modify this list to focus only on specific objects
CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define our target classes (simplified for a more focused demo)
TARGET_CLASSES = ['person', 'bottle', 'cell phone', 'cup', 'laptop', 'chair']

def get_prediction(img, threshold=0.7):
    """Get model predictions and filter based on confidence threshold and target classes"""
    transform = F.to_tensor(img)
    prediction = model([transform])
    
    # Filter predictions by confidence and target classes
    masks = []
    boxes = []
    labels = []
    scores = []
    
    pred_classes = [CLASSES[i] for i in prediction[0]['labels']]
    pred_masks = prediction[0]['masks'].detach().cpu().numpy()
    pred_boxes = prediction[0]['boxes'].detach().cpu().numpy()
    pred_scores = prediction[0]['scores'].detach().cpu().numpy()
    
    for i, score in enumerate(pred_scores):
        if score > threshold and pred_classes[i] in TARGET_CLASSES:
            masks.append(pred_masks[i][0])
            boxes.append(pred_boxes[i])
            labels.append(pred_classes[i])
            scores.append(score)
            
    return masks, boxes, labels, scores

def random_color():
    """Generate a random vibrant color for better visual impact"""
    return COLORS[random.randint(0, len(COLORS)-1)]

def visualize(img, masks, boxes, labels, scores):
    """Apply visually appealing overlays to the image"""
    height, width = img.shape[:2]
    alpha = 0.5  # Transparency factor for mask overlay
    
    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
        color = COLORS[i % len(COLORS)]
        
        # Apply color mask with transparency
        mask_bin = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        colored_mask[mask_bin == 1] = color
        img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
        
        # Draw bounding box with fancy parameters
        x1, y1, x2, y2 = box.astype(int)
        thickness = max(1, int(score * 3))  # Thicker lines for higher confidence
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add fancy label with confidence
        label_text = f"{label}: {score:.2f}"
        label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1 = max(y1, label_size[1])
        
        # Draw label background
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text in white
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add effect: Draw contour of mask
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)
    
    # Add a fancy title
    cv2.putText(img, "Mask R-CNN Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add detected class counts
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    y_pos = 60
    for cls, count in class_counts.items():
        text = f"{cls}: {count}"
        cv2.putText(img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
    
    return img

# Load a pre-trained Mask R-CNN model
print("Loading Mask R-CNN model...")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

# Start video capture
print("Starting webcam...")
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Set properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # FPS calculation removed
    
    # Get predictions
    with torch.no_grad():
        masks, boxes, labels, scores = get_prediction(frame)
    
    # Visualize results
    result = visualize(frame, masks, boxes, labels, scores)
    
    # Show the result
    cv2.imshow('Mask R-CNN Real-time Object Detection', result)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Demo finished!")