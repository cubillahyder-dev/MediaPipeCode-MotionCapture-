import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords, draw_status_box, draw_info_box, is_pose_prone
import time

class PlankDetector:
    def __init__(self):
        self.counter = 0 # Seconds held?
        self.start_time = None
        self.is_holding = False
        self.mp_pose = mp.solutions.pose
        self.is_ready = False

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Check Posture - Prone
        if not is_pose_prone(landmarks):
            draw_info_box(image, "Please Lie Down", color=(0,0,255))
            self.is_ready = False
            # Don't return, as we want to show "Bad Form" or "Not Ready"
            # But the logic below calculates plank time.
        else:
            self.is_ready = True
            draw_info_box(image, "Ready", color=(0,255,0))

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
        # Also require general prone posture
        valid_posture = angle > 160 and angle < 200 and self.is_ready

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

        draw_status_box(image, "PLANK TIME", str(self.counter) + " s")

        cv2.putText(image, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return image
