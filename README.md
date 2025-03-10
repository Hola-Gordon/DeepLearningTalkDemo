# Deep Learning Talk Demo

# Mask R-CNN Real-time Object Detection

A real-time object detection application using PyTorch's pre-trained Mask R-CNN model with OpenCV for webcam integration.

## Features

- Real-time object detection and instance segmentation
- Focused detection of specific objects (people, bottles, cell phones, cups, laptops, chairs)
- Colorful visualization with transparent masks
- Bounding boxes with confidence scores
- Object contour highlighting
- Live count of detected objects by class

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision 0.8+
- OpenCV 4.2+
- NumPy

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/mask-rcnn-detection.git
cd mask-rcnn-detection
```

2. Install dependencies:
```
pip install torch torchvision opencv-python numpy
```

## Usage

Run the script with:
```
python object_detection.py
```

### Controls
- Press 'q' to quit the application
  - Note: Make sure the window has focus when pressing the key
  - You may need to click on the video window first

## Customization

- Modify the `TARGET_CLASSES` list to detect different objects
- Adjust the confidence threshold (default: 0.7) in the `get_prediction` function call
- Change the color scheme by modifying the `COLORS` list
- Adjust webcam resolution by changing the `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT` values

## How It Works

1. The script initializes a pre-trained Mask R-CNN model from PyTorch
2. Each frame from the webcam is processed through the model
3. Predictions are filtered based on confidence threshold and target classes
4. The results are visualized on the frame with masks, bounding boxes, and labels
5. The processed frame is displayed in real-time

## Performance Notes

- GPU acceleration is used when available for faster inference
- You can adjust the frame resolution for better performance on slower systems
- Consider reducing the number of target classes for improved performance