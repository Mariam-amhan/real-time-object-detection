
# Real-Time Object Detection

## Overview

A real-time object detection application built with Python, YOLOv8, and OpenCV.

The application uses a pre-trained YOLOv8n model to detect four object classes — **Person, Dog, Cat, and Car** — from a live webcam feed. It displays bounding boxes, class labels, confidence scores, per-class object counts, and FPS information.

The project also includes multi-threaded frame capture and a bounded frame queue to help manage frame buffering during real-time inference.

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Python Threading and Queue
- PyTorch (CPU/CUDA device selection)

## Features

- Real-time object detection using a pre-trained YOLOv8n model
- Detection of people, dogs, cats, and cars
- Bounding boxes with class labels and confidence scores
- Per-frame object counting by category
- Input and output FPS monitoring
- Multi-threaded camera capture
- Bounded frame queue with a maximum size of two frames
- Frame-dropping mechanism to limit the buildup of outdated frames
- Webcam support and configurable RTSP streaming
- CPU execution and CUDA acceleration when available on supported platforms

## How It Works

1. OpenCV initializes the webcam or configured RTSP video stream.
2. A dedicated capture thread continuously reads incoming frames.
3. Captured frames are placed into a bounded queue.
4. When the queue is full, the capture thread removes the oldest queued frame before attempting to add a new one.
5. The main processing loop retrieves a frame and resizes it to 640 × 480.
6. YOLOv8 performs inference on the frame.
7. Detections are filtered to include only Person, Dog, Cat, and Car.
8. Bounding boxes, confidence scores, and per-class counts are displayed.
9. The annotated frame is displayed along with input and output FPS information.

## Requirements

- Python 3.x
- Webcam or configured RTSP video source
- Dependencies: Ultralytics, OpenCV, and PyTorch

## Installation

Install the required packages:

```bash
pip install ultralytics opencv-python
```

## Run

```bash
python main.py
```

Or:

```bash
python3 main.py
```

Use `python` or `python3` depending on your environment.

Press **ESC** to exit the application.

## Technical Challenges

### Multi-threaded Frame Capture

Camera capture runs in a separate thread from the inference loop, allowing frame acquisition and model processing to operate independently.

### Frame Buffering

A bounded queue with a maximum size of two frames is used to limit buffering. When the queue becomes full, the capture thread removes the oldest queued frame before attempting to insert the latest frame.

If the queue fills again before the new frame can be inserted, the insertion is skipped rather than blocking the capture thread.

This approach is intended to reduce the accumulation of outdated frames, although end-to-end latency has not yet been benchmarked.

### Device Selection

The application uses CPU inference on macOS and selects CUDA on Windows/Linux when PyTorch detects an available CUDA device. Otherwise, it falls back to CPU execution.

### Detection Filtering

The YOLOv8 model's detections are filtered to four selected classes, keeping the displayed results focused on the intended use case.

## Notes

- Object counts represent detections within the current frame, not unique objects tracked across multiple frames.
- RTSP streaming requires a valid camera URL and compatible video capture configuration.
- Detection accuracy and processing performance depend on the model, hardware, lighting, and video quality.
- FPS values are displayed for monitoring, but end-to-end capture-to-display latency is not currently measured.


## Future Improvements

- Add a graphical user interface (GUI)
- Support video file uploads
- Save detection results to files
- Implement object tracking with persistent IDs
- Measure end-to-end capture-to-display latency
- Compare different queue sizes and frame-dropping strategies, including keeping only the latest frame
- Separate the detection and display logic more clearly using a producer-consumer architecture
- Benchmark FP16 (half-precision) inference on compatible CUDA GPUs to evaluate potential FPS improvements
- Improve frame-buffering and performance monitoring


## Feedback

Suggestions and contributions are welcome, particularly regarding real-time inference performance, frame buffering, and code architecture.
