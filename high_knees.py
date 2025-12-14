import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility

class HighKneeDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose
        self.last_leg = None # "left" or "right"

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count high knees.
        Counts one rep for each knee raise or a pair? Usually each.
        """
        h, w, _ = image.shape

        # Check both knees
        l_hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        l_knee_idx = self.mp_pose.PoseLandmark.LEFT_KNEE.value
        r_hip_idx = self.mp_pose.PoseLandmark.RIGHT_HIP.value
        r_knee_idx = self.mp_pose.PoseLandmark.RIGHT_KNEE.value

        # Visibility
        if not check_visibility(landmarks, [l_hip_idx, l_knee_idx, r_hip_idx, r_knee_idx]):
            return image

        l_hip_y = landmarks[l_hip_idx].y
        l_knee_y = landmarks[l_knee_idx].y
        r_hip_y = landmarks[r_hip_idx].y
        r_knee_y = landmarks[r_knee_idx].y

        # High Knee Logic
        # Knee should come up close to hip level.
        # Threshold: Knee Y < Hip Y + offset (since Y increases downwards)
        # Actually if Knee Y < Hip Y, knee is ABOVE hip.
        # Let's say Knee Y < Hip Y + 0.1 (normalized) allows slightly below hip.

        # We need a state machine that tracks which leg is up.

        # Check Left Leg
        left_up = l_knee_y < (l_hip_y + 0.05)
        # Check Right Leg
        right_up = r_knee_y < (r_hip_y + 0.05)

        if left_up and self.last_leg != 'left':
            self.counter += 1
            self.last_leg = 'left'
            print(f"High Knee count: {self.counter}")
        elif right_up and self.last_leg != 'right':
            self.counter += 1
            self.last_leg = 'right'
            print(f"High Knee count: {self.counter}")

        # Reset if both down?
        # Actually High Knees are rapid.
        # If neither is up, reset last_leg?
        # If I do L-R-L-R.
        # L up -> count, last=L.
        # L down, R up -> count, last=R.
        # This works.

        # UI
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'LEG', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.last_leg),
                    (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return image
