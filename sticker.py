import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import gradio as gr

# Set up color for sticker border (white)
BORDER_COLOR = (255, 255, 255)

# COCO classes (used by Mask R-CNN). Note: some entries are marked as 'N/A'
# as the COCO dataset does not include a class for those indices.
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

# We are focusing on the person class for sticker creation.
TARGET_CLASSES = ['person']

def get_prediction(img, threshold=0.7):
    """
    Get model predictions filtered for person detection.
    
    Parameters:
    - img: Input image in BGR format.
    - threshold: Score threshold to filter weak predictions.
    
    Returns:
    - masks: List of masks for detected persons.
    - boxes: Bounding boxes for detected persons.
    - labels: Class labels for each detected object (only persons in our case).
    - scores: Confidence scores for each detection.
    """
    # Convert image to tensor expected by the model
    transform = F.to_tensor(img)
    
    # Get predictions from the model
    prediction = model([transform])
    
    # Initialize lists to store filtered predictions
    masks = []
    boxes = []
    labels = []
    scores = []
    
    # Convert model output labels to human-readable classes
    pred_classes = [CLASSES[i] for i in prediction[0]['labels']]
    # Get masks, boxes, and scores from the prediction
    pred_masks = prediction[0]['masks'].detach().cpu().numpy()
    pred_boxes = prediction[0]['boxes'].detach().cpu().numpy()
    pred_scores = prediction[0]['scores'].detach().cpu().numpy()
    
    # Filter detections based on threshold and target class (person)
    for i, score in enumerate(pred_scores):
        if score > threshold and pred_classes[i] in TARGET_CLASSES:
            masks.append(pred_masks[i][0])
            boxes.append(pred_boxes[i])
            labels.append(pred_classes[i])
            scores.append(score)
            
    return masks, boxes, labels, scores

def create_sticker(img, mask, border_thickness=3):
    """
    Create a sticker image by applying the mask to the image.
    The area outside the mask is made transparent and a border is drawn.
    
    Parameters:
    - img: Input image in BGR format.
    - mask: Mask corresponding to the object (person).
    - border_thickness: Thickness of the white border around the mask.
    
    Returns:
    - sticker: The resulting sticker image with transparency (BGRA format).
    """
    # Create a binary mask where mask values > 0.5 become 1 and others 0.
    mask_bin = (mask > 0.5).astype(np.uint8)
    
    # Convert the image to BGRA (BGR + Alpha channel)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = (mask_bin * 255).astype(np.uint8)
    sticker = cv2.merge([b_channel, g_channel, r_channel, alpha_channel])
    
    # Find contours on the binary mask to create a border
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Draw the border on the sticker image
        cv2.drawContours(sticker, [cnt], -1, BORDER_COLOR, border_thickness)
    
    return sticker

def process_image(input_img):
    """
    Process the uploaded image:
      - Run detection for persons.
      - If found:
          * If one person is detected, create a sticker using that mask.
          * If multiple persons are detected, combine their masks and then create a sticker.
      - If no person is detected, return the original image with a notification.
    
    Parameters:
    - input_img: The image provided by the user (numpy array).
    
    Returns:
    - Processed image: Either a sticker with transparency or the original image with a notification.
    """
    # Ensure image is in BGR format. If image has 4 channels (BGRA), convert to BGR.
    if input_img.shape[2] == 4:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        masks, boxes, labels, scores = get_prediction(input_img)
    
    # Check if any persons were detected
    if len(masks) == 0:
        # If no person is detected, add a notification to the image.
        print("No person detected")
        output_img = input_img.copy()
        cv2.putText(output_img, "No person detected.", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)
        
        # Optionally, display a separate notification window (this is optional and might not work in some environments)
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, "No person detected.", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return output_img
    elif len(masks) == 1:
        # If only one person is detected, create a sticker from the single mask.
        sticker = create_sticker(input_img, masks[0])
    else:
        # If multiple persons are detected, combine their masks via pixel-wise maximum.
        combined_mask = np.zeros_like(masks[0])
        for m in masks:
            combined_mask = np.maximum(combined_mask, m)
        sticker = create_sticker(input_img, combined_mask)
    
    return sticker

# Load the pre-trained Mask R-CNN model.
# The model is downloaded and set to evaluation mode.
print("Loading Mask R-CNN model...")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Check if GPU is available and move the model to GPU for faster inference.
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

# Create a Gradio interface with an image upload widget.
description = """
Upload an image containing one or more persons. The model will detect the person(s) and convert them into a sticker with a transparent background and a white border.
"""

# Define the Gradio interface.
iface = gr.Interface(
    fn=process_image,  # Function to process the image.
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Sticker Output"),
    title="Person Sticker Maker",
    description=description,
    allow_flagging="never"
)

# Launch the Gradio app.
iface.launch()
