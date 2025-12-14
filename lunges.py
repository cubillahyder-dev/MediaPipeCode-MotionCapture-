import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility

class LungeDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count lunges.
        """
        h, w, _ = image.shape

        # Use left leg for detection (assuming side view)
        hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        knee_idx = self.mp_pose.PoseLandmark.LEFT_KNEE.value
        ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        # Check visibility
        if not check_visibility(landmarks, [hip_idx, knee_idx, ankle_idx]):
            return image

        # Get coordinates
        l_hip = [landmarks[hip_idx].x * w, landmarks[hip_idx].y * h]
        l_knee = [landmarks[knee_idx].x * w, landmarks[knee_idx].y * h]
        l_ankle = [landmarks[ankle_idx].x * w, landmarks[ankle_idx].y * h]

        # Calculate knee angle
        angle = calculate_angle(l_hip, l_knee, l_ankle)

        # Visualize angle
        cv2.putText(image, str(int(angle)),
                           tuple(np.array(l_knee).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Lunge Logic
        # Up (Standing): ~170-180
        # Down (Lunge): ~90

        if angle > 160:
            if self.stage == 'down':
                self.counter += 1
                print(f"Lunge count: {self.counter}")
            self.stage = "up"
        if angle < 100:
            self.stage = "down"

        # UI
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage),
                    (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
