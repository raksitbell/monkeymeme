# MonkeyMeme Gesture & Head Controller

This project uses MediaPipe and OpenCV to recognize hand gestures, detect head movements, and display corresponding images.

## Features

- **Hand Gesture Recognition**: Detects Fist, Index Finger, Tilted Index Finger, and Default (Open Hand).
- **Head Pose Detection**: Detects head direction (Left, Right, Up, Down, Forward).
- **Face Framing**: Draws a bounding box around the detected face.
- **Single Window Interface**: Displays webcam feed and gesture image side-by-side.

## Prerequisites

- Python 3.7+
- Webcam

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install opencv-python mediapipe numpy
    ```

## Usage

1.  **Run the script**:

    ```bash
    python main.py
    ```

2.  **Controls**:

    - **Quit**: Press **'q'**.

3.  **Interactions**:
    - **Hand Gestures**: Show specific hand signs to change the displayed image.
    - **Head Movement**: Look Left, Right, Up, or Down to see the direction detected on screen.
    - **Face Box**: A green box will track your face.

## Troubleshooting

- **Webcam not opening**: Ensure no other application is using the camera.
- **Images not loading**: Make sure `1.png`, `2.png`, `3.png`, and `4.png` are in the same directory as `main.py`.
