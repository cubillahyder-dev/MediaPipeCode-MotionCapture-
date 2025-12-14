# AI Fitness Trainer

This repository contains a computer vision-based fitness trainer that tracks and counts repetitions for various exercises using Google's MediaPipe Pose estimation.

## Exercises Supported

- **Push-ups**: Counts repetitions based on elbow angle.
- **Squats**: Counts repetitions based on knee angle.
- **Ab Crunches**: Counts repetitions based on hip curl.
- **Plank**: specific timer that runs only when proper form (straight back) is maintained.

## Project Structure

- `pushups.py`: Logic for Push-up detection.
- `squats.py`: Logic for Squat detection.
- `crunches.py`: Logic for Ab Crunch detection.
- `plank.py`: Logic for Plank form check and timer.
- `pose_utils.py`: Utility functions (angle calculation).
- `demo.ipynb`: Jupyter Notebook to run and test the exercises.
- `requirements.txt`: List of dependencies.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: You need `opencv-python`, `mediapipe`, and `numpy`.*

## How to Run

1.  Open the `demo.ipynb` file in Jupyter Notebook or JupyterLab.
2.  In the third cell, change the `EXERCISE` variable to the exercise you want to perform:
    - `'pushup'`
    - `'squat'`
    - `'crunch'`
    - `'plank'`
3.  Run the cells. A window named 'AI Fitness Trainer' will pop up showing your webcam feed with the overlay.
4.  Perform the exercise. The counters/timers will update automatically.
5.  Press `q` on your keyboard to quit the application.

## Troubleshooting

-   **Webcam not opening**: Ensure no other application is using the camera. Check if `cv2.VideoCapture(0)` index is correct for your system (try 1 or 2 if 0 fails).
-   **Dependencies**: If you encounter errors, make sure you are using a Python environment compatible with MediaPipe (Python 3.8-3.11 recommended).
