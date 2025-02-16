# YOLO Character Detection - Modern Family Edition

This repository contains a Python implementation of YOLO object detection using OpenCV's DNN module. The goal of this project is to train a YOLO model to recognize characters from the TV series **Modern Family**.

## Features
- Train a YOLOv8 model to detect Modern Family characters
- Export the trained model to ONNX format
- Perform object detection on images and video streams
- Display bounding boxes with class labels

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/modernfamilyYOLO.git
   cd modernfamilyYOLO
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dependencies
- OpenCV
- NumPy
- PyYAML
- Ultralytics (for YOLOv8 training and exporting)

## Training the Model
To train the YOLOv8 model on the Modern Family dataset, run `train.py`:
```python
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model using the dataset specified in config.yaml
results = model.train(data="config.yaml", epochs=100)
```

## Exporting the Model to ONNX
After training, export the best model for inference (`export.py`): 
```python
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

## Running Object Detection on an Image
```python
from yolo_predictions import YOLO_pred
import cv2

test = YOLO_pred('runs/detect/train/weights/best.onnx', 'config.yaml')
result = test.predictions("image.jpg")
cv2.imshow("prediction", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Running Real-time Object Detection on a Video
```python
test.real_time_prediction("video.mp4")
```

## Example 
The program detects Modern Family characters in an image or video and displays them with bounding boxes and class labels.

![Example Output](example.jpg)

## Based On
This project was inspired by the following tutorials:
- [YOLO Object Detection Using OpenCV And Python](https://www.youtube.com/watch?v=mRhQmRm_egc)
- [Train Yolov8 object detection on a custom dataset](https://www.youtube.com/watch?v=m9fH9OWn8YM)

## Author
[Gonçalo Oliveira](https://github.com/goncalOtree)













