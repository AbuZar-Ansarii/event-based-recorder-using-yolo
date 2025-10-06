import cv2
import os
import time
import json
import random
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# -----------------------------------------------------------
# Configuration 
# -----------------------------------------------------------
PRE_SECONDS = 15               # seconds to keep in buffer (before event)
POST_SECONDS = 15              # seconds to save after event
DEFAULT_FPS = 20               # fallback FPS if camera doesn’t report one
VIDEO_DIR = "videos"           # directory to save clips
METADATA_FILE = "metadata.json"
MODEL_NAME = "yolov8n.pt"      # small, fast YOLOv8 model
CONF_THRESH = 0.4              # confidence threshold for detection
DETECT_EVERY_N_FRAMES = 2      # run YOLO every N frames (reduce CPU load)
TRIGGER_CLASSES = ["person", "car", "dog", "cat", "bicycle"]  # objects that trigger saving

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

def ensure_directories():
    """Create output folders and files if they don’t exist."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "w") as f:
            json.dump([], f, indent=2)

def random_gps():
    """Simulate GPS coordinates for demo purposes."""
    return {
        "lat": round(random.uniform(-90, 90), 6),
        "lon": round(random.uniform(-180, 180), 6)
    }

def current_timestamp():
    """Return current UTC timestamp for filenames."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def save_event_metadata(event_type, filepath):
    """Append a new event entry to metadata.json."""
    event = {
        "event_type": event_type,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "gps": random_gps(),
        "file_path": filepath
    }
    try:
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(event)
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_fps(cap):
    """Try to read FPS from camera, fallback if unavailable."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    return int(fps) if fps > 1 else DEFAULT_FPS

def draw_detections(frame, detections, class_names):
    """
    Draw bounding boxes and labels for each detected object.
    """
    for box in detections:
        cls_id = int(box.cls[0])             # class ID
        conf = float(box.conf[0])            # confidence
        label = f"{class_names[cls_id]} {conf:.2f}"

        # Get coordinates of the box (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0)  # green box

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# -----------------------------------------------------------
# Main Application
# -----------------------------------------------------------

def main(camera_source="0"):
    """
    Start the YOLO-based event recorder.
    Args:
        camera_source: webcam index (0) or IP camera URL (e.g. "http://192.168.x.x:8080/video")
    """
    ensure_directories()

    # Try converting source to int 
    try:
        source = int(camera_source)
    except ValueError:
        source = camera_source  # IP Webcam URL

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera source: {camera_source}")
        return

    # Basic camera setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    frame_size = (width, height)
    fps = get_fps(cap)
    print(f"Camera connected at {frame_size}, FPS={fps}")
    print(f"Triggering on classes: {TRIGGER_CLASSES}")

    # Load YOLOv8 model 
    model = YOLO(MODEL_NAME)
    class_names = model.names  # mapping of class IDs to names

    # Rolling buffer: stores the last N frames in memory
    buffer = deque(maxlen=int(PRE_SECONDS * fps))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    detected_counter = 0       # how many frames detection persists
    persist_required = 3       # must persist 3 cycles to trigger

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        # Add frame to buffer (rolling 15s)
        buffer.append(frame)
        display_frame = frame.copy()

        # Run YOLO every N frames to reduce CPU usage
        if len(buffer) % DETECT_EVERY_N_FRAMES == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)
            detections = results[0].boxes
            display_frame = draw_detections(display_frame, detections, class_names)

            # Collect detected class names in this frame
            detected_labels = [class_names[int(b.cls[0])] for b in detections]

            # Check if any trigger class is present
            if any(lbl in TRIGGER_CLASSES for lbl in detected_labels):
                detected_counter += 1
            else:
                detected_counter = max(0, detected_counter - 1)

            # If object persists for enough frames, trigger event
            if detected_counter >= persist_required:
                timestamp = current_timestamp()
                filename = f"event_{timestamp}.mp4"
                filepath = os.path.join(VIDEO_DIR, filename)
                print(f"\n Event triggered: {detected_labels}")
                print(f"Saving video clip -> {filepath}")

                # Create video writer
                writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

                # Write pre-event buffered frames
                for buffered_frame in buffer:
                    writer.write(buffered_frame)

                # Continue writing for POST_SECONDS after detection
                end_time = time.time() + POST_SECONDS
                while time.time() < end_time:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break

                    # Run YOLO again to draw boxes during saving
                    results2 = model(frame2, conf=CONF_THRESH, verbose=False)
                    detections2 = results2[0].boxes
                    frame2_with_boxes = draw_detections(frame2, detections2, class_names)

                    writer.write(frame2_with_boxes)
                    cv2.imshow("Recording Event...", frame2_with_boxes)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


                writer.release()
                save_event_metadata("yolo_detection", filepath)
                print(f"Clip saved and metadata updated.\n")
                detected_counter = 0  # reset after event

        # Show live detection feed
        cv2.imshow("YOLOv8 Live Detection", display_frame)

        # Quit if user presses 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting program.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("All resources released. Goodbye!")

# -----------------------------------------------------------
# Entry Point
# -----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Event-Based Video Recorder")
    parser.add_argument(
        "--camera",
        type=str,
        default="http://100.109.99.83:8080/video",
        help="IP Webcam URL"
    )

    args = parser.parse_args()
    main(args.camera)
