import av
import cv2
import numpy as np
from ultralytics import YOLO
import json

def extract_frames(video_path):
    container = av.open(video_path)
    frames = []
    timestamps = []

    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format='bgr24'))
        timestamps.append(frame.time)

    return frames, timestamps

def calculate_distance_from_center(frame, x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    distance = np.sqrt((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2)
    return distance

def draw_detections(frames, results, model):
    for frame, result in zip(frames, results):
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Unpack the bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = box.cls[0]  # Class label index

                # Get the class name from the model's names dictionary
                class_name = model.names[int(cls)]

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"No boxes detected in frame")

    return frames

def detect_logos(frames, model):
    results = []
    for frame in frames:
        result = model.predict(source=frame)[0]  # Get the first (and only) result
        results.append(result)
    return results

def save_annotations(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()

def save_detections_to_json(timestamps, results, model, json_path):
    detections = {
        "Pepsi_pts": [],
        "CocaCola_pts": []
    }
    for timestamp, result in zip(timestamps, results):
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Unpack the bounding box coordinates
                cls = box.cls[0]  # Class label index
                class_name = model.names[int(cls)]
                width = x2 - x1
                height = y2 - y1
                distance = calculate_distance_from_center(result.orig_img, x1, y1, x2, y2)
                
                detection_info = {
                    "timestamp": round(timestamp, 2),  # Rounding off timestamp
                    "width (pixels)": round(float(width), 2),  # Rounding off width
                    "height (pixels)": round(float(height), 2),  # Rounding off height
                    "distance_from_center (pixels)": round(float(distance), 2)  # Rounding off distance
                }
                
                if class_name == "Pepsi":
                    detections["Pepsi_pts"].append(detection_info)
                elif class_name == "CocaCola":
                    detections["CocaCola_pts"].append(detection_info)

    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=4)

# Example usage
video_path = 'video3.mp4'
model_path = 'y.pt'  # Updated to use YOLOv8x
output_video_path = 'video3_detections.mp4'
output_json_path = 'detections_3.json'

# Load the YOLO model
model = YOLO(model_path)

# Extract frames and timestamps from the video
frames, timestamps = extract_frames(video_path)

# Detect logos in the frames
results = detect_logos(frames, model)

# Annotate frames with detections
annotated_frames = draw_detections(frames, results, model)

# Save the annotated frames as a video
save_annotations(annotated_frames, output_video_path)

# Save the detections to a JSON file
save_detections_to_json(timestamps, results, model, output_json_path)
