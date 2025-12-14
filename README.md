# AI Fitness Trainer

This repository contains a computer vision-based fitness trainer that tracks and counts repetitions for various exercises using Google's MediaPipe Pose estimation.

## Exercises Supported

- **Push-ups**: Counts repetitions based on elbow angle.
- **Squats**: Counts repetitions based on knee angle.
- **Ab Crunches**: Counts repetitions based on hip curl.
- **Plank**: Specific timer that runs only when proper form (straight back) is maintained.
- **Lunges**: Counts repetitions based on knee angle.
- **Bicep Curls**: Counts repetitions based on elbow angle.
- **Jumping Jacks**: Counts repetitions based on arm movement (wrists above shoulders).
- **High Knees**: Counts repetitions based on knee height relative to hip.
- **Punches**: Counts repetitions based on arm extension (jab/cross).

## Project Structure

- `pushups.py`: Logic for Push-up detection.
- `squats.py`: Logic for Squat detection.
- `crunches.py`: Logic for Ab Crunch detection.
- `plank.py`: Logic for Plank form check and timer.
- `lunges.py`: Logic for Lunge detection.
- `bicep_curls.py`: Logic for Bicep Curl detection.
- `jumping_jacks.py`: Logic for Jumping Jack detection.
- `high_knees.py`: Logic for High Knee detection.
- `punches.py`: Logic for Punch detection.
- `pose_utils.py`: Utility functions (angle calculation, visibility check).
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
    - `'lunge'`
    - `'bicep_curl'`
    - `'jumping_jack'`
    - `'high_knee'`
    - `'punch'`
3.  Run the cells. A window named 'AI Fitness Trainer' will pop up showing your webcam feed with the overlay.
4.  Perform the exercise. The counters/timers will update automatically.
5.  Press `q` on your keyboard to quit the application.

## Troubleshooting

-   **Webcam not opening**: Ensure no other application is using the camera. Check if `cv2.VideoCapture(0)` index is correct for your system (try 1 or 2 if 0 fails).
-   **Dependencies**: If you encounter errors, make sure you are using a Python environment compatible with MediaPipe (Python 3.8-3.11 recommended).
