import cv2
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import numpy as np


def speech(text):
    print(text)
    language = 'en'
    p = "assets/sounds/output.mp3"
    output = gTTS(text=text, lang=language, slow=False)
    output.save(p)
    playsound(p)


def draw_bbox(image, boxes, labels, confidences, colors=None):
    """
    Draw bounding boxes on image
    """
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, (box, label, conf) in enumerate(zip(boxes, labels, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label and confidence
        label_text = f"{label}: {conf:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


# Load YOLO model
model = YOLO('yolov8x.pt')

video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)  # verbose=False to reduce console output

    # Extract results
    boxes = []
    detected_labels = []
    confidences = []

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])

                # Get confidence
                conf = box.conf[0].cpu().numpy()
                confidences.append(conf)

                # Get class name
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls_id]
                detected_labels.append(class_name)

    # Draw bounding boxes
    output_image = draw_bbox(frame.copy(), boxes, detected_labels, confidences)

    cv2.imshow("Object Detection", output_image)

    # Add new labels to the list
    for item in detected_labels:
        if item not in labels:
            labels.append(item)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Create speech output
if labels:
    new_sentence = 'I found'
    for i, label in enumerate(labels):
        if i == len(labels) - 1 and len(labels) > 1:
            new_sentence += f' and a {label}'
        else:
            new_sentence += f' a {label},'

    # Remove trailing comma and add period
    new_sentence = new_sentence.rstrip(',') + '.'
    speech(new_sentence)
else:
    speech("No objects detected.")