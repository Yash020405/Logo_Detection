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
                x1, y1, x2, y2 = box.xyxy[0] 
                conf = box.conf[0]
                cls = box.cls[0]

                class_name = model.names[int(cls)]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"No boxes detected in frame")

    return frames

def detect_logos(frames, model):
    results = []
    for frame in frames:
        result = model.predict(source=frame)[0]
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
                x1, y1, x2, y2 = box.xyxy[0] 
                cls = box.cls[0]  
                class_name = model.names[int(cls)]
                width = x2 - x1
                height = y2 - y1
                distance = calculate_distance_from_center(result.orig_img, x1, y1, x2, y2)
                
                detection_info = {
                    "timestamp": round(timestamp, 2),
                    "width (pixels)": round(float(width), 2),
                    "height (pixels)": round(float(height), 2),
                    "distance_from_center (pixels)": round(float(distance), 2)
                }
                
                if class_name == "Pepsi":
                    detections["Pepsi_pts"].append(detection_info)
                elif class_name == "CocaCola":
                    detections["CocaCola_pts"].append(detection_info)

    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=4)

video_path = 'path_to_video.mp4' # Path to the video file
model_path = 'Model/best.pt' 
output_video_path = 'annotated_video.mp4'
output_json_path = 'results.json'

model = YOLO(model_path)

frames, timestamps = extract_frames(video_path)

results = detect_logos(frames, model)

annotated_frames = draw_detections(frames, results, model)

save_annotations(annotated_frames, output_video_path)

save_detections_to_json(timestamps, results, model, output_json_path)
