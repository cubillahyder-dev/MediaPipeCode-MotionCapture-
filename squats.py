import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle

class SquatDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None # "up" or "down"
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count squats.
        """
        h, w, _ = image.shape

        # Get coordinates for left leg
        l_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        l_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        l_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        # Calculate knee angle
        angle = calculate_angle(l_hip, l_knee, l_ankle)

        # Visualize angle
        cv2.putText(image, str(int(angle)),
                           tuple(np.array(l_knee).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Squat Logic
        if angle > 160:
            if self.stage == 'down':
                self.counter += 1
                print(f"Squat count: {self.counter}")
            self.stage = "up"
        if angle < 90:
            self.stage = "down"

        # Draw UI box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage),
                    (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
