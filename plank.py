import cv2
import numpy as np
import mediapipe as mp
import time
from pose_utils import calculate_angle

class PlankTimer:
    def __init__(self):
        self.counter = 0 # This will act as seconds
        self.stage = None # "good" or "bad" form
        self.mp_pose = mp.solutions.pose
        self.start_time = None
        self.total_time = 0

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to timer for plank.
        """
        h, w, _ = image.shape

        # Check for plank form: straight line from shoulder to heel.
        l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        l_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        l_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        l_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

        hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

        # Visualize angles
        cv2.putText(image, f"Hip: {int(hip_angle)}",
                           tuple(np.array(l_hip).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, f"Knee: {int(knee_angle)}",
                           tuple(np.array(l_knee).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Plank Logic
        good_form = (160 < hip_angle < 190) and (160 < knee_angle < 190)

        if good_form:
            self.stage = "good"
            if self.start_time is None:
                self.start_time = time.time()

            current_time = time.time()
            self.total_time += (current_time - self.start_time)
            self.start_time = current_time # Update start time to avoid double counting if breaks happen
        else:
            self.stage = "bad"
            self.start_time = None # Stop timer

        # Draw UI box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # Time data
        cv2.putText(image, 'TIME', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(self.total_time)),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (85,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage),
                    (80,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
