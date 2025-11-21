import cv2
import mediapipe as mp
import numpy as np
import math
import time
from typing import Dict, Optional, Tuple, Any

class GestureController:
    """
    Controls gesture recognition, head pose detection, and image display logic.
    """
    def __init__(self, display_size: Tuple[int, int] = (600, 400)):
        """
        Initialize the GestureController.

        Args:
            display_size: Tuple (width, height) for the display window.
        """
        self.display_size = display_size
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Image resources
        self.image_gallery: Dict[str, np.ndarray] = {}
        self.current_display_image: Optional[np.ndarray] = None
        
        # Gesture state
        self.last_gesture = "default"
        self.gesture_cooldown = 0
        self.COOLDOWN_FRAMES = 15
        
        self.load_resources()

    def load_resources(self) -> None:
        """
        Loads and resizes images for the gallery from disk.
        Exits the program if images are missing.
        """
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

            # Create unknown placeholder (black image with text)
            unknown_img = np.zeros(self.display_size + (3,), dtype=np.uint8)
            cv2.putText(unknown_img, "Unknown", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.image_gallery["unknown"] = unknown_img
            
            self.current_display_image = self.image_gallery["default"]
            print("Resources loaded successfully.")

        except Exception as e:
            print(f"!!! Error loading resources: {e}")
            exit(1)

    def _is_finger_folded(self, tip, pip) -> bool:
        """
        Checks if a finger is folded based on tip and pip y-coordinates.
        In screen coordinates, y increases downwards.
        So if tip.y > pip.y, the tip is below the pip, meaning folded (for an upright hand).
        """
        return tip.y > pip.y

    def detect_gesture(self, landmarks: Any) -> str:
        """
        Analyzes hand landmarks to determine the current gesture.

        Args:
            landmarks: MediaPipe hand landmarks object.

        Returns:
            str: The name of the detected gesture ('fist', 'index_finger', 'tilted', 'default').
        """
        # Extract key landmarks
        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]

        # Check state of each finger (except thumb, which is less reliable for these simple gestures)
        is_index_folded = self._is_finger_folded(index_tip, index_pip)
        is_middle_folded = self._is_finger_folded(middle_tip, middle_pip)
        is_ring_folded = self._is_finger_folded(ring_tip, ring_pip)
        is_pinky_folded = self._is_finger_folded(pinky_tip, pinky_pip)

        # 1. Fist Detection
        # All four fingers (Index, Middle, Ring, Pinky) are folded.
        if is_index_folded and is_middle_folded and is_ring_folded and is_pinky_folded:
            return "fist"

        # 2. Index Finger Detection (Straight or Tilted)
        # Index is extended (NOT folded), others are folded.
        
        # More robust check for index extension: Tip is further from wrist than PIP is.
        dist_wrist_tip = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
        dist_wrist_pip = math.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y)
        is_index_extended = dist_wrist_tip > dist_wrist_pip

        if is_index_extended and is_middle_folded and is_ring_folded and is_pinky_folded:
            # Differentiate between Straight and Tilted based on X-axis deviation
            # Calculate absolute difference in X between wrist and index tip.
            x_difference = abs(index_tip.x - wrist.x)
            
            # Threshold of 0.15 determined experimentally.
            if x_difference > 0.15:
                return "tilted"
            else:
                return "index_finger"
        
        return "default"

    def detect_head_pose(self, image: np.ndarray, face_landmarks: Any) -> Tuple[str, np.ndarray]:
        """
        Estimates head pose using PnP algorithm.
        Returns the direction (Left, Right, Up, Down, Forward) and the nose tip 2D coordinates.
        """
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        # Landmarks indices for PnP
        # Nose tip: 1
        # Chin: 199
        # Left eye left corner: 33
        # Right eye right corner: 263
        # Left Mouth corner: 61
        # Right mouth corner: 291
        landmark_indices = [1, 199, 33, 263, 61, 291]

        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in landmark_indices:
                if idx == 1: # Nose tip
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])       
        
        # Convert to NumPy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # Distortion matrix
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        # Determine direction
        if y < -10:
            text = "Looking Left"
        elif y > 10:
            text = "Looking Right"
        elif x < -10:
            text = "Looking Down"
        elif x > 10:
            text = "Looking Up"
        else:
            text = "Forward"

        return text, nose_2d

    def draw_face_box(self, image: np.ndarray, face_landmarks: Any) -> None:
        """Draws a bounding box around the detected face."""
        img_h, img_w, _ = image.shape
        x_min, y_min = img_w, img_h
        x_max, y_max = 0, 0

        for lm in face_landmarks.landmark:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    def update_display(self, gesture: str) -> None:
        """
        Updates the displayed image based on the detected gesture.
        Implements a cooldown to prevent flickering between gestures.
        """
        if gesture != self.last_gesture and self.gesture_cooldown == 0:
            print(f"Gesture detected: {gesture}")
            self.current_display_image = self.image_gallery.get(gesture, self.image_gallery["default"])
            self.last_gesture = gesture
            self.gesture_cooldown = self.COOLDOWN_FRAMES
        
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1

    def run(self) -> None:
        """
        Main application loop.
        Handles video capture, processing, and display.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print(f"System Ready. Display Size: {self.display_size}")
        print("Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame. Retrying...")
                    time.sleep(0.1)
                    continue

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, self.display_size)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process Hands
                hand_results = self.hands.process(rgb_frame)
                
                # Process Face
                face_results = self.face_mesh.process(rgb_frame)
                
                current_gesture = "default"
                
                # Hand Processing
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        current_gesture = self.detect_gesture(hand_landmarks)
                
                # Face Processing
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Draw Bounding Box
                        self.draw_face_box(frame, face_landmarks)
                        
                        # Detect Head Pose
                        try:
                            direction, nose_2d = self.detect_head_pose(frame, face_landmarks)
                            
                            # Display Direction
                            cv2.putText(frame, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Project Nose (Optional visual)
                            p1 = (int(nose_2d[0]), int(nose_2d[1]))
                            p2 = (int(nose_2d[0] + 0), int(nose_2d[1] - 50)) # Just a simple line up for now
                            # cv2.line(frame, p1, p2, (255, 0, 0), 3)

                        except Exception as e:
                            pass # PnP might fail in some edge cases

                self.update_display(current_gesture)

                # Combine Webcam Frame and Gesture Image
                if self.current_display_image is not None:
                    combined_display = np.hstack((frame, self.current_display_image))
                else:
                    combined_display = frame

                # Show Window
                cv2.imshow("MonkeyMeme Controller", combined_display)

                # Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                        
        finally:
            # Cleanup resources
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.face_mesh.close()

if __name__ == "__main__":
    app = GestureController()
    app.run()