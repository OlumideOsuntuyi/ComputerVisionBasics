import cv2
import numpy as np
import face_recognition
import pickle
import pygame
import os
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
from pathlib import Path

OWNER = 'Olumide George Osuntuyi'


def speech(text):
    print(text)
    language = 'en'

    # Set path relative to the repo root
    output_dir = Path(__file__).parent / "assets" / "sounds"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create if not exists

    output_path = output_dir / "output.mp3"

    # Generate and save audio
    output = gTTS(text=text, lang=language, slow=False)
    output.save(str(output_path))

    pygame.mixer.init()
    pygame.mixer.music.load(output_path)
    pygame.mixer.music.play()

    # Wait for it to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()


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


def draw_face_bbox(image, face_locations, face_names):
    """
    Draw bounding boxes around faces with names
    """
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw rectangle around face
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 2)

        # Draw label background
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (255, 0, 255), cv2.FILLED)

        # Draw name
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return image


def train_face_recognition_model():
    """
    Train face recognition model from dataset and save it
    """
    print("Training face recognition model...")

    known_face_encodings = []
    known_face_names = []

    dataset_path = "./assets/dataset/face-recognition"

    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found: {dataset_path}")
        return

    # Create models directory if it doesn't exist
    os.makedirs("./models/face-recognition", exist_ok=True)

    # Process each person's folder
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)

        if not os.path.isdir(person_path):
            continue

        # Convert folder name to display name (replace - with spaces)
        person_name = person_folder.replace('-', ' ')
        print(f"Processing {person_name}...")

        # Process each image in the person's folder
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(person_path, filename)

                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)

                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        # Use the first face found in the image
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        print(f"  Added {filename}")
                    else:
                        print(f"  No face found in {filename}")

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

    # Save the model
    model_data = {
        'encodings': known_face_encodings,
        'names': known_face_names
    }

    model_path = "./models/face-recognition/current.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Face recognition model saved to {model_path}")
    print(f"Total faces trained: {len(known_face_encodings)}")
    print(f"People recognized: {list(set(known_face_names))}")


def load_face_recognition_model():
    """
    Load the trained face recognition model
    """
    model_path = "./models/face-recognition/current.pkl"

    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Please run train_face_recognition_model() first")
        return [], []

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        print(f"Loaded face recognition model with {len(model_data['encodings'])} faces")
        return model_data['encodings'], model_data['names']

    except Exception as e:
        print(f"Error loading face recognition model: {e}")
        return [], []


def recognize_faces_and_objects():
    """
    Main function to recognize faces and detect objects in real-time
    """
    # Load models
    print("Loading YOLO model...")
    yolo_model = YOLO('yolo11n.pt')

    print("Loading face recognition model...")
    known_face_encodings, known_face_names = load_face_recognition_model()

    if not known_face_encodings:
        print("No face recognition model available. Please train the model first.")
        return

    # Start video capture
    video = cv2.VideoCapture(0)

    detected_objects = []
    detected_people = []

    # Process every other frame for better performance
    process_this_frame = 0
    delay = 2

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame == 0:
            # Face recognition
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                # Add to detected people list
                if name not in detected_people and name != "Unknown":
                    detected_people.append(name)
                    speech(f'Found {name}')

            # Object detection with YOLO
            results = yolo_model(frame, verbose=False)

            # Extract YOLO results
            yolo_boxes = []
            yolo_labels = []
            yolo_confidences = []

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        yolo_boxes.append([x1, y1, x2, y2])

                        conf = box.conf[0].cpu().numpy()
                        yolo_confidences.append(conf)

                        cls_id = int(box.cls[0].cpu().numpy())
                        class_name = yolo_model.names[cls_id]
                        yolo_labels.append(class_name)

                        # Add to detected objects (exclude person class since we handle faces separately)
                        if class_name not in detected_objects and class_name != "person":
                            detected_objects.append(class_name)

        process_this_frame = (process_this_frame + 1) % delay

        # Scale back up face locations
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]

        # Draw face bounding boxes
        frame = draw_face_bbox(frame, face_locations, face_names)

        # Draw object bounding boxes
        if process_this_frame:  # Only draw when we have fresh results
            frame = draw_bbox(frame, yolo_boxes, yolo_labels, yolo_confidences)

        # Display the frame
        cv2.imshow("Face Recognition + Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    # Create speech output with emphasis on people
    speech_text = ""

    if detected_people:
        if len(detected_people) == 1:
            speech_text = f"I can see {detected_people[0]}"
        else:
            speech_text = f"I can see {', '.join(detected_people[:-1])} and {detected_people[-1]}"

    if detected_objects:
        if speech_text:
            speech_text += f". I also found {', '.join(detected_objects)}"
        else:
            speech_text = f"I found {', '.join(detected_objects)}"

    if not speech_text:
        speech_text = "I don't see any recognizable people or objects"

    speech_text += "."
    speech(speech_text)


# Example usage:
if __name__ == "__main__":
    # First, train the model (run this once when you add new people)
    train_face_recognition_model()

    # Then run the recognition system
    recognize_faces_and_objects()