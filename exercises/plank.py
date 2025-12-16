import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords
import time

class PlankDetector:
    def __init__(self):
        self.counter = 0 # Seconds held?
        self.start_time = None
        self.is_holding = False
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Plank: Body straight, horizontal.
        # Shoulders, Hips, Ankles should be roughly on a line.
        # Elbows under shoulders.

        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        l_hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        l_ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        if not check_visibility(landmarks, [l_shoulder_idx, l_hip_idx, l_ankle_idx]):
            return image

        l_shoulder = get_landmark_coords(landmarks, l_shoulder_idx, w, h)
        l_hip = get_landmark_coords(landmarks, l_hip_idx, w, h)
        l_ankle = get_landmark_coords(landmarks, l_ankle_idx, w, h)

        # Calculate angle at hip (should be ~180)
        angle = calculate_angle(l_shoulder, l_hip, l_ankle)

        cv2.putText(image, str(int(angle)),
                   tuple(np.array(l_hip).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Logic
        # Angle should be between say 170 and 190 (straight body)
        # Also, the body should be horizontal. Check slope between shoulder and ankle?
        # Slope = (y2-y1)/(x2-x1). If horizontal, y diff is small.
        # But if camera is top down?
        # Let's assume side view.

        # Tolerance for hip angle
        valid_posture = angle > 160 and angle < 200

        if valid_posture:
            if not self.is_holding:
                self.start_time = time.time()
                self.is_holding = True

            current_time = time.time()
            self.counter = int(current_time - self.start_time)
            status = "HOLDING"
            color = (0,255,0)
        else:
            self.is_holding = False
            self.start_time = None
            status = "BAD FORM"
            color = (0,0,255)
            # Retain last counter or reset? Usually plank is max time.
            # Let's just show current hold time.

        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(image, 'PLANK TIME', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter) + " s",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return image
