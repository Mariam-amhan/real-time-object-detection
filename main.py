import cv2
import time
import threading
import queue
from datetime import datetime

from ultralytics import YOLO
import platform

try:
    import torch
except ImportError:
    torch = None

# ==========================
# CONFIG
# ==========================

USE_RTSP = False
RTSP_URL = 'rtsp://admin:admin4867@192.168.60.27:554/cam/realmonitor?channel=1&subtype=0'

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_QUEUE_SIZE = 2
SHOW_FPS_ON_FRAME = True
PRINT_DETECTIONS = False

USE_OPENVINO = False
OPENVINO_MODEL_PATH = "yolov8n_openvino_model/"

# Detect if running on macOS
IS_MAC = platform.system() == "Darwin"

# ==========================
# CAMERA CAPTURE THREAD
# ==========================

class CameraCapture(threading.Thread):
    def __init__(self, frame_queue, use_rtsp=False, rtsp_url=None, cam_index=0):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.use_rtsp = use_rtsp
        self.rtsp_url = rtsp_url
        self.cam_index = cam_index
        self.stop_event = threading.Event()
        self.capture = None
        self.input_fps = 0.0
        self._last_time = time.time()

    def open_camera(self):

        if self.use_rtsp:
            print("[Capture] Opening RTSP stream...")

            # macOS requires ffmpeg support for RTSP
            self.capture = cv2.VideoCapture(self.rtsp_url)

        else:
            print("[Capture] Opening webcam...")

            if IS_MAC:
                # macOS uses AVFoundation
                self.capture = cv2.VideoCapture(self.cam_index, cv2.CAP_AVFOUNDATION)
            else:
                # Windows / Linux
                self.capture = cv2.VideoCapture(self.cam_index)

        if not self.capture.isOpened():
            print("❌ [Capture] Could not open camera/stream.")
            return False

        print("✅ [Capture] Camera/stream opened.")
        return True

    def run(self):
        if not self.open_camera():
            return

        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                print("⚠️ [Capture] Failed to read frame.")
                time.sleep(0.01)
                continue

            # Update input FPS
            now = time.time()
            dt = now - self._last_time
            if dt > 0:
                self.input_fps = 0.9 * self.input_fps + 0.1 * (1.0 / dt) if self.input_fps > 0 else 1.0 / dt
            self._last_time = now

            # Drop older frames
            if not self.frame_queue.empty() and self.frame_queue.qsize() >= MAX_QUEUE_SIZE:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        self.capture.release()
        print("[Capture] Stopped.")

    def stop(self):
        self.stop_event.set()


# ==========================
# MODEL LOADING
# ==========================

def load_model():
    model_path = "yolov8n.pt"

    if USE_OPENVINO:
        print("[Model] Loading OpenVINO model...")
        model = YOLO(OPENVINO_MODEL_PATH)
        return model, "openvino"

    print("[Model] Loading YOLO model:", model_path)
    model = YOLO(model_path)

    # macOS = CPU only
    if IS_MAC:
        device = "cpu"
    else:
        # Windows/Linux
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    print(f"[Model] Using device: {device}")
    model.to(device)
    return model, device


# ==========================
# MAIN LOOP
# ==========================

def main():
    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    capture_thread = CameraCapture(
        frame_queue,
        use_rtsp=USE_RTSP,
        rtsp_url=RTSP_URL,
        cam_index=CAM_INDEX
    )
    capture_thread.start()

    model, device = load_model()

    output_fps = 0.0
    last_infer_time = time.time()

    print("Press ESC to exit.")

    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
        except queue.Empty:
            print("⚠️ [Main] No frame received from capture thread.")
            continue

        # Resize
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Inference
        t0 = time.time()
        results = model(frame_resized, verbose=False)[0]
        t1 = time.time()
        # ==========================
        # FILTER BOXES (ONLY WANTED CLASSES)
        # ==========================

        ALLOWED_CLASSES = ["person", "dog", "cat", "car"]

        filtered_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            if class_name in ALLOWED_CLASSES:
                filtered_boxes.append(box)


        results.boxes = filtered_boxes

        # ==========================
        # FILTERED OBJECT COUNTING
        # ==========================

        object_counts = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            if class_name in ALLOWED_CLASSES:
                if class_name not in object_counts:
                    object_counts[class_name] = 1
                else:
                    object_counts[class_name] += 1

        dt = t1 - last_infer_time
        if dt > 0:
            output_fps = 0.9 * output_fps + 0.1 * (1.0 / dt) if output_fps > 0 else 1.0 / dt
        last_infer_time = t1

        annotated_frame = results.plot()

        y_offset = 90
        for obj, count in object_counts.items():
            cv2.putText(
                annotated_frame,
                f"{obj}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            y_offset += 30

        if SHOW_FPS_ON_FRAME:
            in_fps = capture_thread.input_fps
            cv2.putText(annotated_frame, f"Input FPS:  {in_fps:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Output FPS: {output_fps:.2f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if PRINT_DETECTIONS:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = results.names[cls_id]
                print(f"[{timestamp}] {name} ({conf:.2f})")

        cv2.imshow("YOLO - Real-Time", annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    capture_thread.stop()
    capture_thread.join()
    cv2.destroyAllWindows()
    print("✅ Exited cleanly.")


if __name__ == "__main__":
    ALLOWED_CLASSES = ["person", "dog", "cat", "car"]

    main()