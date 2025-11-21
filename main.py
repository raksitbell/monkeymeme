import cv2
import mediapipe as mp
import numpy as np
import math
import time
from typing import Dict, Optional, Tuple, Any

class GestureController:
    """
    Controls gesture recognition and image display logic.
    """
    def __init__(self, display_size: Tuple[int, int] = (600, 400)):
        self.display_size = display_size
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.image_gallery: Dict[str, np.ndarray] = {}
        self.current_display_image: Optional[np.ndarray] = None
        
        self.last_gesture = "default"
        self.gesture_cooldown = 0
        self.COOLDOWN_FRAMES = 15
        
        self.load_resources()

    def load_resources(self) -> None:
        """Loads and resizes images for the gallery."""
        try:
            # Load images
            images = {
                "fist": cv2.imread("1.png"),
                "index_finger": cv2.imread("2.png"),
                "default": cv2.imread("3.png"),
                "tilted": cv2.imread("4.png")
            }
            
            # Resize and validate
            for key, img in images.items():
                if img is None:
                    raise FileNotFoundError(f"Could not load image for '{key}'. Check file path.")
                self.image_gallery[key] = cv2.resize(img, self.display_size)

            # Create unknown placeholder
            unknown_img = np.zeros(self.display_size + (3,), dtype=np.uint8)
            cv2.putText(unknown_img, "Unknown", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.image_gallery["unknown"] = unknown_img
            
            self.current_display_image = self.image_gallery["default"]
            print("Resources loaded successfully.")

        except Exception as e:
            print(f"!!! Error loading resources: {e}")
            exit(1)

    def detect_gesture(self, landmarks: Any) -> str:
        """
        Analyzes hand landmarks to determine the current gesture.
        """
        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]

        # Check folded fingers (tip below pip in y-axis for upright hand)
        # Note: In screen coordinates, y increases downwards.
        is_middle_folded = middle_tip.y > middle_pip.y
        is_ring_folded = ring_tip.y > ring_pip.y
        is_pinky_folded = pinky_tip.y > pinky_pip.y

        # 1. Fist: Index also folded
        if (index_tip.y > index_pip.y) and is_middle_folded and is_ring_folded and is_pinky_folded:
            return "fist"

        # 2. & 3. Index Finger (Straight or Tilted)
        # Check if index is extended (tip further from wrist than pip)
        dist_wrist_tip = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
        dist_wrist_pip = math.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y)
        is_index_extended = dist_wrist_tip > dist_wrist_pip

        if is_index_extended and is_middle_folded and is_ring_folded and is_pinky_folded:
            # Check tilt based on x-difference between wrist and index tip
            x_difference = abs(index_tip.x - wrist.x)
            if x_difference > 0.15:
                return "tilted"
            else:
                return "index_finger"
        
        return "default"

    def update_display(self, gesture: str) -> None:
        """Updates the displayed image based on the detected gesture."""
        if gesture != self.last_gesture and self.gesture_cooldown == 0:
            print(f"Gesture detected: {gesture}")
            self.current_display_image = self.image_gallery.get(gesture, self.image_gallery["default"])
            self.last_gesture = gesture
            self.gesture_cooldown = self.COOLDOWN_FRAMES
        
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1

    def run(self) -> None:
        """Main loop for video capture and processing."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print(f"System Ready. Display Size: {self.display_size}")
        
        prev_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip and Resize
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, self.display_size)
                
                # Process Hand
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                current_gesture = "default"
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        current_gesture = self.detect_gesture(hand_landmarks)
                
                self.update_display(current_gesture)

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show Windows
                cv2.imshow("Webcam", frame)
                if self.current_display_image is not None:
                    cv2.imshow("Display Image", self.current_display_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    app = GestureController()
    app.run()