import cv2
import mediapipe as mp
import numpy as np

# Import detectors
from exercises.shadow_boxing import ShadowBoxingDetector
from exercises.pushups import PushUpDetector
from exercises.squats import SquatDetector
from exercises.mountain_climbers import MountainClimberDetector
from exercises.jumping_jacks import JumpingJackDetector
from exercises.plank import PlankDetector
from exercises.burpees import BurpeeDetector

def main():
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Select Exercise
    # For demo purposes, we could cycle through or pick one.
    # Let's prompt or set a default.
    print("Available Exercises:")
    print("1. Shadow Boxing")
    print("2. Push-ups")
    print("3. Squats")
    print("4. Mountain Climbers")
    print("5. Jumping Jacks")
    print("6. Plank")
    print("7. Burpees")

    # Defaulting to 1 for this script, but in a notebook user can choose class.
    # To make this useful, let's create a dictionary.

    detectors = {
        '1': ShadowBoxingDetector(),
        '2': PushUpDetector(),
        '3': SquatDetector(),
        '4': MountainClimberDetector(),
        '5': JumpingJackDetector(),
        '6': PlankDetector(),
        '7': BurpeeDetector()
    }

    choice = input("Select exercise (1-7): ")
    if choice not in detectors:
        print("Invalid choice")
        return

    detector = detectors[choice]

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the camera feed
            frame = cv2.flip(frame, 1)

            # Recolor to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Process specific exercise
                image = detector.process(image, landmarks)

            except Exception as e:
                # print(e)
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
