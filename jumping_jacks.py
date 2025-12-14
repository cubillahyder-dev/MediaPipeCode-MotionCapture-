import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility

class JumpingJackDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count jumping jacks.
        """
        h, w, _ = image.shape

        # Check hands (Shoulder-Elbow-Wrist isn't enough, we need Shoulder-Hip-Wrist? No)
        # Hands go above head.
        # Check Wrist Y relative to Shoulder Y.

        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value
        r_wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

        # Check legs (Hip-Knee-Ankle doesn't change much, but Hip-Knee or Hip-Ankle distance change?)
        # Or Hip-Hip (width)? No.
        # Hip abduction. Angle between spine and leg?
        # Simpler: Wrist position.

        # Visibility check
        if not check_visibility(landmarks, [l_shoulder_idx, r_shoulder_idx, l_wrist_idx, r_wrist_idx]):
            return image

        l_shoulder_y = landmarks[l_shoulder_idx].y
        l_wrist_y = landmarks[l_wrist_idx].y

        # Logic
        # Down (Hands down): Wrist Y > Shoulder Y
        # Up (Hands up): Wrist Y < Shoulder Y (significantly)

        # Using normalized coordinates for simple height check

        if l_wrist_y > l_shoulder_y:
            self.stage = "down"
        if l_wrist_y < l_shoulder_y and self.stage == 'down':
            self.stage = "up"
            self.counter += 1
            print(f"Jumping Jack count: {self.counter}")

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
