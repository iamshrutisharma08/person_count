from flask import Flask, Response, jsonify
import cv2
import argparse
import threading
from ultralytics import YOLO
import supervision as sv
from fer import FER


# Initialize Flask app
app = Flask(__name__)


# Global variables for capturing frames
global cap
cap=cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None


def parse_arguments() -> argparse.Namespace:
    parser= argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", 
                        default= [1280,720],
                          nargs=2, 
                          type=int)
    
    args = parser.parse_args()
    return args

def main():
    args= parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    emotion_detector = FER(mtcnn=True)

    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width )
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    


    model= YOLO("yolov8l.pt")

    # Initialize Emotion Detector
    emotion_detector = FER(mtcnn=True)
   
# Function to process each frame with YOLO and FER
def process_frame(frame):
    results = model(frame)
    happy_count = 0
    not_happy_count = 0
    total_count = 0

    # Box annotator for YOLO
    box_annotator = sv.BoxAnnotator(thickness=2)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Filter for "person" class (ID=0 for COCO dataset)
        person_class_id = 0
        person_indices = class_ids == person_class_id
        person_boxes = boxes[person_indices]

        labels = []
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]  # Crop face region

            if face.size > 0:
                # Perform emotion detection
                emotions = emotion_detector.top_emotion(face)
                if emotions is not None:
                    emotion, score = emotions
                    if emotion == "happy":
                        happy_count += 1
                    else:
                        not_happy_count += 1
                    labels.append(f"{emotion} {score:.2f}")
                else:
                    labels.append("Unknown")

        # Update total counts
        total_count = happy_count + not_happy_count

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences[person_indices],
            class_id=class_ids[person_indices],
        )

        # Add labels to detections
        detections.labels = labels

        # Annotate the frame
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

        # Overlay the counts
        overlay_text = f"Total: {total_count} | Happy: {happy_count} | Not Happy: {not_happy_count}"
        cv2.putText(
            annotated_frame,
            overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return annotated_frame

# Flask route to serve the live video feed
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is None:
                    continue
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()

            # Yield frame to the client
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for processing
@app.route('/process', methods=['GET'])
def process():
    global current_frame
    with frame_lock:
        if current_frame is None:
            return jsonify({"error": "No frame available for processing"}), 400

        # Process the current frame
        processed_frame = process_frame(current_frame)
        return jsonify({"message": "Frame processed successfully"}), 200


# Thread to capture frames from the webcam
def capture_frames():
    global cap 
    global current_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            current_frame = frame

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

        

       
