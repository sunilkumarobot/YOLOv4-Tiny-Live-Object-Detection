# YOLOv4-Tiny Object Detection using OpenCV

## Project Overview

This project demonstrates object detection using the YOLOv4-Tiny deep learning model and OpenCV's DNN (Deep Neural Network) module.

The system detects multiple objects in an input image, draws bounding boxes around detected objects, and displays the corresponding class labels along with confidence scores.

YOLO (You Only Look Once) is one of the most popular real-time object detection algorithms because it performs object localization and classification in a single forward pass of the neural network.

YOLOv4-Tiny is a lightweight version of YOLOv4 that provides faster inference while maintaining good detection accuracy, making it suitable for real-time and resource-constrained applications.

---

## Features

* Object detection on images
* YOLOv4-Tiny implementation using OpenCV DNN
* Bounding box generation
* Class label prediction
* Confidence score display
* Non-Maximum Suppression (NMS)
* Fast and lightweight inference

---

## Technologies Used

* Python
* OpenCV
* YOLOv4-Tiny
* NumPy

---

## Project Workflow

```text
Input Image
      ↓
Image Preprocessing
      ↓
Blob Creation
      ↓
YOLOv4-Tiny Network
      ↓
Object Detection
      ↓
Confidence Filtering
      ↓
Non-Maximum Suppression
      ↓
Final Bounding Boxes
      ↓
Output Image
```

---

## Model Files

The project uses the following YOLOv4-Tiny files:

### yolov4-tiny.weights

Contains the trained weights learned by the neural network.

### yolov4-tiny.cfg

Contains the network architecture and configuration parameters.

### coco.names

Contains the list of object classes that can be detected by the model.

Examples:

* Person
* Car
* Bicycle
* Dog
* Cat
* Chair
* Bottle
* Laptop
* Cell Phone
* Traffic Light

and many more.

---

## Important Concepts

### Objectness Score

Objectness score indicates the probability that a bounding box contains an object.

Example:

```text
Objectness = 0.95
```

This means the network is 95% confident that an object exists inside the predicted bounding box.

---

### Confidence Score

Confidence score indicates how confident the network is about the predicted class.

Example:

```text
Person = 0.92
Chair = 0.03
Bottle = 0.01
```

The object will be classified as **Person** because it has the highest confidence score.

---

### Non-Maximum Suppression (NMS)

NMS removes duplicate overlapping bounding boxes and keeps only the box with the highest confidence score.

Example:

Before NMS:

```text
Car Box 1 = 95%
Car Box 2 = 91%
Car Box 3 = 88%
```

After NMS:

```text
Keep Car Box 1
Remove Car Box 2
Remove Car Box 3
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/YOLOv4-Tiny-Object-Detection.git
```

Move into the project directory:

```bash
cd YOLOv4-Tiny-Object-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Execute:

```bash
python yolo4.py
```

The model will process the input image and display the detected objects with bounding boxes and confidence scores.

---

## Detection Result

### Input Image

Add your original image here.

### Output Image

Add your detection result image here.

Example:

```markdown
![Detection Result](images/traffic_detection.jpg)
```

---

## Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning based Object Detection
* YOLOv4-Tiny Architecture
* OpenCV DNN Module
* Blob Creation
* Forward Propagation
* Confidence Thresholding
* Objectness Thresholding
* Non-Maximum Suppression (NMS)
* Bounding Box Visualization

---

## Future Improvements

* Live Webcam Object Detection
* YOLOv8 Implementation
* Object Tracking using DeepSORT
* ROS2 Integration
* Real-Time Robotics Vision Applications

---

## Author

**Sunil Kumar**

Robotics Engineer | Computer Vision Enthusiast | OpenCV Developer

---

### Output image
<img width="272" height="374" alt="image" src="https://github.com/user-attachments/assets/14c27b03-8390-4fe8-af8c-4a6ef1926e49" />
