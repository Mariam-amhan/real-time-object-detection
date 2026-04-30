# Real-Time Object Detection

## Overview
A real-time object detection system built using Python, YOLOv8, and OpenCV.

This project detects objects from a live webcam or video stream in real time.  
It identifies the object type, draws bounding boxes around detected objects, displays the confidence score, counts detected objects, and shows the FPS performance on the screen.

## Tech Stack
- Python
- YOLOv8
- OpenCV

## Features
- Real-time object detection
- Detects multiple object types
- Draws bounding boxes around detected objects
- Displays object labels and confidence scores
- Counts the number of detected objects
- Supports live webcam or RTSP video stream
- Displays FPS for performance monitoring

## How It Works
1. The camera or video stream is opened using OpenCV.
2. Each video frame is passed to the YOLOv8 model.
3. The model detects objects in the frame.
4. Detected objects are labeled with their class name and confidence score.
5. Bounding boxes are drawn around the detected objects.
6. The total number of detected objects is displayed.
7. The processed frame is shown in real time.

## Requirements
* Python 3.x
* Webcam or RTSP stream

## Setup
```bash
pip install ultralytics opencv-python
```

## Run
```bash
python main.py
# or
python3 main.py
```

Use `python` or `python3` depending on your system.

## Notes
* Make sure a webcam or RTSP source is available.
* Press ESC to exit the application.
* The detection accuracy depends on the YOLOv8 model and video quality.

## Future Improvements
* Add a graphical user interface
* Save detection results to a file
* Add support for uploading video files
* Improve object counting by category
