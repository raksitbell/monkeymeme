import cv2
import mediapipe as mp
import numpy as np
import math


DISPLAY_SIZE = (600, 400)

try:
    image_fist = cv2.imread("1.png")           
    image_index_finger = cv2.imread("2.png")   
    image_default = cv2.imread("3.png")        
    image_tilted = cv2.imread("4.png")         

    if image_fist is not None: image_fist = cv2.resize(image_fist, DISPLAY_SIZE)
    if image_index_finger is not None: image_index_finger = cv2.resize(image_index_finger, DISPLAY_SIZE)
    if image_default is not None: image_default = cv2.resize(image_default, DISPLAY_SIZE)
    if image_tilted is not None: image_tilted = cv2.resize(image_tilted, DISPLAY_SIZE)
    
    image_gallery = {
        "default": image_default,
        "index_finger": image_index_finger,
        "fist": image_fist,
        "tilted": image_tilted,
        "unknown": cv2.putText(np.zeros(DISPLAY_SIZE + (3,), dtype=np.uint8), 
                               "Unknown", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    }

    if any(img is None for img in [image_gallery["default"], image_gallery["index_finger"], image_gallery["fist"], image_gallery["tilted"]]):
        print("!!! Error: Cannot load images.")
        print("Please check '1.png', '2.png', '3.png', and '4.png'.")
        exit()

except Exception as e:
    print(f"!!! Error: {e}")
    exit()

current_display_image = image_gallery["default"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("System Ready.")
print("Both windows are now sized:", DISPLAY_SIZE)

def check_gesture(landmarks):
    wrist = landmarks.landmark[0]
    index_tip = landmarks.landmark[8]
    index_pip = landmarks.landmark[6]
    middle_tip = landmarks.landmark[12]
    middle_pip = landmarks.landmark[10]
    ring_tip = landmarks.landmark[16]
    ring_pip = landmarks.landmark[14]
    pinky_tip = landmarks.landmark[20]
    pinky_pip = landmarks.landmark[18]

    is_middle_folded = middle_tip.y > middle_pip.y
    is_ring_folded = ring_tip.y > ring_pip.y
    is_pinky_folded = pinky_tip.y > pinky_pip.y

    # 1. ท่ากำมือ
    if (index_tip.y > index_pip.y) and is_middle_folded and is_ring_folded and is_pinky_folded:
        return "fist"

    # 2. & 3. ท่าชูนิ้วชี้ (ตรง/เอียง)
    dist_wrist_tip = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
    dist_wrist_pip = math.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y)
    is_index_really_extended = dist_wrist_tip > dist_wrist_pip

    if is_index_really_extended and is_middle_folded and is_ring_folded and is_pinky_folded:
        x_difference = abs(index_tip.x - wrist.x)
        if x_difference > 0.15:
            return "tilted"
        else:
            return "index_finger"
    
    return "default"

last_gesture = "default"
gesture_cooldown = 0
COOLDOWN_FRAMES = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    frame = cv2.resize(frame, DISPLAY_SIZE) 

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "default"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_gesture = check_gesture(hand_landmarks)

    if current_gesture != last_gesture and gesture_cooldown == 0:
        print(f"Gesture: {current_gesture}")
        if current_gesture == "index_finger":
            current_display_image = image_gallery["index_finger"]
        elif current_gesture == "tilted":
            current_display_image = image_gallery["tilted"]
        elif current_gesture == "fist":
            current_display_image = image_gallery["fist"]
        else:
            current_display_image = image_gallery["default"]
            
        last_gesture = current_gesture
        gesture_cooldown = COOLDOWN_FRAMES 
    
    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    cv2.imshow("Webcam", frame)
    cv2.imshow("Display Image", current_display_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()