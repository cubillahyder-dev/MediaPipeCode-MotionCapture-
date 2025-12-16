import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords

class JumpingJackDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None # 'close' or 'open'
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Jumping jacks:
        # Hands go above head.
        # Legs go wide.

        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        l_wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value
        l_hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        l_ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        r_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        r_wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value
        r_hip_idx = self.mp_pose.PoseLandmark.RIGHT_HIP.value
        r_ankle_idx = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value

        if not check_visibility(landmarks, [l_shoulder_idx, l_wrist_idx, r_shoulder_idx, r_wrist_idx]):
            return image

        l_wrist = get_landmark_coords(landmarks, l_wrist_idx, w, h)
        r_wrist = get_landmark_coords(landmarks, r_wrist_idx, w, h)
        l_shoulder = get_landmark_coords(landmarks, l_shoulder_idx, w, h)
        r_shoulder = get_landmark_coords(landmarks, r_shoulder_idx, w, h)

        # Simple check: Wrists above shoulders (Y coordinate is smaller)
        hands_up = (l_wrist[1] < l_shoulder[1]) and (r_wrist[1] < r_shoulder[1])
        hands_down = (l_wrist[1] > l_shoulder[1]) and (r_wrist[1] > r_shoulder[1])

        # We can also check leg width, but hands are usually sufficient.

        if hands_up:
            self.stage = "up"

        if hands_down and self.stage == 'up':
            self.stage = "down"
            self.counter += 1

        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(image, 'JUMP JACKS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
