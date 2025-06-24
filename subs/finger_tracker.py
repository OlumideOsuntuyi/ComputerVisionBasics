import threading
import time
import pyautogui

import cv2
import mediapipe as mp
import numpy as np
import math
from gtts import gTTS
import pygame

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

from pathlib import Path

running = True
speech_text = ''
next_speech_text = ''

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

def clamp01(x:float):
    return 0 if x < 0 else 1 if x > 1 else x

def move_mouse_to_normalized(x_norm, y_norm):
    screen_width, screen_height = pyautogui.size()
    x = int(x_norm * screen_width)
    y = int(y_norm * screen_height)
    pyautogui.moveTo(x, y, duration=0)


class FingerTracker:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Finger tip and PIP landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.pip_ids = [3, 6, 10, 14, 18]  # PIP joints for finger detection

        # Gesture recognition
        self.gesture_history = []
        self.gesture_threshold = 10

        # Colors for visualization
        self.colors = {
            'thumb': (255, 0, 0),  # Red
            'index': (0, 255, 0),  # Green
            'middle': (0, 0, 255),  # Blue
            'ring': (255, 255, 0),  # Yellow
            'pinky': (255, 0, 255)  # Magenta
        }

        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    @staticmethod
    def speech(text):
        """Text-to-speech function"""
        print(text)
        language = 'en'

        output_dir = Path(__file__).parent / "assets" / "sounds"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "finger_output.mp3"

        try:
            output = gTTS(text=text, lang=language, slow=False)
            output.save(str(output_path))

            pygame.mixer.init()
            pygame.mixer.music.load(output_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.quit()
        except Exception as e:
            print(f"Speech error: {e}")

    def get_finger_positions(self, landmarks):
        """Get positions of all fingertips and joints"""
        positions = {}

        # Get landmark positions
        for i, name in enumerate(self.finger_names):
            tip_id = self.tip_ids[i]
            positions[f'{name.lower()}_tip'] = (
                int(landmarks[tip_id].x * 640),  # Assuming 640 width
                int(landmarks[tip_id].y * 480)  # Assuming 480 height
            )

            if i > 0:  # Skip thumb PIP (different structure)
                pip_id = self.pip_ids[i]
                positions[f'{name.lower()}_pip'] = (
                    int(landmarks[pip_id].x * 640),
                    int(landmarks[pip_id].y * 480)
                )

        return positions

    def detect_fingers_up(self, landmarks):
        """Detect which fingers are extended"""
        fingers_up = []

        # Thumb (compare x coordinates for left/right hand)
        if landmarks[self.tip_ids[0]].x > landmarks[self.tip_ids[0] - 1].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # Other fingers (compare y coordinates)
        for i in range(1, 5):
            if landmarks[self.tip_ids[i]].y < landmarks[self.pip_ids[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        return fingers_up

    def recognize_gesture(self, fingers_up):
        """Recognize hand gestures based on finger positions"""
        gestures = {
            (0, 0, 0, 0, 0): "Fist",
            (1, 1, 1, 1, 1): "Open Hand",
            (0, 1, 0, 0, 0): "Pointing",
            (0, 1, 1, 0, 0): "Peace Sign",
            (1, 0, 0, 0, 0): "Thumbs Up",
            (0, 0, 0, 0, 1): "Pinky Up",
            (1, 1, 0, 0, 1): "Rock On",
            (0, 1, 1, 1, 0): "Three Fingers",
            (0, 1, 1, 1, 1): "Four Fingers"
        }

        fingers_tuple = tuple(fingers_up)
        return gestures.get(fingers_tuple, f"Custom ({sum(fingers_up)} fingers)")

    def calculate_finger_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def detect_finger_interactions(self, positions):
        """Detect finger interactions like pinching"""
        interactions = []

        # Check thumb-index pinch
        if 'thumb_tip' in positions and 'index_tip' in positions:
            distance = self.calculate_finger_distance(
                positions['thumb_tip'],
                positions['index_tip']
            )
            if distance < 30:  # Adjust threshold as needed
                interactions.append("Thumb-Index Pinch")

        # Check other finger combinations
        finger_tips = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                if finger_tips[i] in positions and finger_tips[j] in positions:
                    distance = self.calculate_finger_distance(
                        positions[finger_tips[i]],
                        positions[finger_tips[j]]
                    )
                    if distance < 25:
                        finger1 = finger_tips[i].replace('_tip', '').title()
                        finger2 = finger_tips[j].replace('_tip', '').title()
                        interactions.append(f"{finger1}-{finger2} Touch")

        return interactions

    def draw_finger_tracking(self, image, landmarks, hand_connections):
        """Draw finger tracking visualization"""
        h, w, c = image.shape

        # Draw hand connections
        self.mp_draw.draw_landmarks(
            image, landmarks, hand_connections,
            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

        # Draw fingertips with different colors
        for i, (tip_id, color, name) in enumerate(zip(self.tip_ids, self.colors.values(), self.finger_names)):
            cx, cy = int(landmarks.landmark[tip_id].x * w), int(landmarks.landmark[tip_id].y * h)
            cv2.circle(image, (cx, cy), 8, color, cv2.FILLED)
            cv2.putText(image, name, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return image

    def track_fingers(self):
        """Main finger tracking function"""

        global running, next_speech_text, volume
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        gesture_announced = False
        frame_count = 0

        print("Starting finger tracking... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            frame_count += 1

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand label (Left/Right)
                    hand_label = results.multi_handedness[hand_idx].classification[0].label

                    # Draw hand tracking
                    frame = self.draw_finger_tracking(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # Get finger positions
                    positions = self.get_finger_positions(hand_landmarks.landmark)

                    # Detect which fingers are up
                    fingers_up = self.detect_fingers_up(hand_landmarks.landmark)

                    # Recognize gesture
                    gesture = self.recognize_gesture(fingers_up)

                    # Detect finger interactions
                    interactions = self.detect_finger_interactions(positions)

                    # Display information
                    y_offset = 30 + (hand_idx * 150)
                    cv2.putText(frame, f"{hand_label} Hand:", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.putText(frame, f"Gesture: {gesture}", (10, y_offset + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Display finger status
                    finger_status = ""
                    for i, (finger, is_up) in enumerate(zip(self.finger_names, fingers_up)):
                        status = "UP" if is_up else "DOWN"
                        finger_status += f"{finger}: {status}  "

                    cv2.putText(frame, finger_status, (10, y_offset + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Display interactions
                    if interactions:
                        interaction_text = ", ".join(interactions)
                        cv2.putText(frame, f"Interactions: {interaction_text}",
                                    (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


                    index_pos: tuple[float, float] = positions['index_tip']
                    xnorm, ynorm = (index_pos[0] / 640.0, index_pos[1] / 480.0)

                    thumb_index_distance = self.calculate_finger_distance(positions['thumb_tip'], index_pos)

                    volume.SetMasterVolumeLevelScalar(clamp01(thumb_index_distance / 250.0), None)
                    #pyautogui.moveRel(0, (thumb_index_distance / 250) * 50, duration=1)
                    cv2.putText(frame, f"Finger Distance: {thumb_index_distance}",
                                (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 200), 1)

                    if frame_count % 2 == 0 and gesture != "Custom (0 fingers)":
                        move_mouse_to_normalized(xnorm, ynorm)

                    # Announce gesture every 30 frames (about 1 second)
                    if frame_count % 30 == 0 and gesture != "Custom (0 fingers)":
                        if hand_idx == 0:  # Only announce for first hand
                            next_speech_text = f"{hand_label} hand showing {gesture}"

            else:
                cv2.putText(frame, "No hands detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Finger Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        running = False
        cap.release()
        cv2.destroyAllWindows()

    def analyze_static_image(self, image_path):
        """Analyze finger positions in a static image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[hand_idx].classification[0].label

                # Draw tracking
                image = self.draw_finger_tracking(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Analyze fingers
                fingers_up = self.detect_fingers_up(hand_landmarks.landmark)
                gesture = self.recognize_gesture(fingers_up)

                print(f"{hand_label} Hand - Gesture: {gesture}")
                print(f"Fingers up: {fingers_up}")

        cv2.imshow('Finger Analysis', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_speech():
    global running, speech_text, next_speech_text
    while running:
        if len(speech_text) > 0:
            FingerTracker.speech(text=speech_text)
            speech_text = next_speech_text
            next_speech_text = ''
        else:
            speech_text = next_speech_text
            time.sleep(0.5)


def main():
    """Main function to run finger tracking"""
    tracker = FingerTracker()

    print("Finger Tracking Options:")
    print("1. Real-time tracking (default)")
    print("2. Analyze static image")

    choice = '1' #input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        image_path = input("Enter image path: ").strip()
        tracker.analyze_static_image(image_path)
    else:
        tracker.track_fingers()


#threading.Thread(daemon=True, target=run_speech).start()
main()