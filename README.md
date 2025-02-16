# YOLO Character Detection - Modern Family Edition

This repository contains a Python implementation of YOLO object detection using OpenCV's DNN module. The goal of this project was to train a YOLO model to recognize characters from the TV series **Modern Family**.

![Example Output](example.gif)

## Features
- Train a YOLOv8 model to detect Modern Family characters
- Export the trained model to ONNX format
- Perform object detection on images and video streams
- Display bounding boxes with class labels

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/goncalOtree/modernfamilyYOLO.git
   cd modernfamilyYOLO
   ```

## Dependencies
- OpenCV
- NumPy
- PyYAML
- Ultralytics [optional] (for YOLOv8 training and exporting)

## Model Files
- `model.onnx` - The trained YOLO model in ONNX format.
- `config.yaml` - Configuration file containing class names and other settings.

## Running Object Detection on an Image
```python
from yolo_predictions import YOLO_pred
import cv2

test = YOLO_pred('model.onnx', 'config.yaml')
result = test.predictions("image.jpg")
cv2.imshow("prediction", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Running Real-time Object Detection on a Video
```python
test.real_time_prediction("video.mp4")
```

## Training and Exporting the Model
- **Training:** The model was trained using `train.py` with a YOLOv8 model.
- **Exporting:** The trained model was exported to ONNX format using `export.py`.

## How It Works
- **Model Inference:** Uses OpenCV’s DNN module to run inference with `model.onnx`.
- **Image Preprocessing:** Resizes and normalizes input images.
- **Post-processing:** Filters predictions using Non-Maximum Suppression (NMS).
- **Visualization:** Draws bounding boxes and class labels on the image.

## Additional Files
- `playground.ipynb` - A Jupyter Notebook demonstrating example usage.
- `not_used_on_training/` - A directory containing test images that were not part of the training dataset.

## Based On
This project was inspired by the following tutorials:
- [YOLO Object Detection Using OpenCV And Python](https://www.youtube.com/watch?v=mRhQmRm_egc)
- [Train Yolov8 object detection on a custom dataset](https://www.youtube.com/watch?v=m9fH9OWn8YM)

## Author
[Gonçalo Oliveira](https://github.com/goncalOtree)














