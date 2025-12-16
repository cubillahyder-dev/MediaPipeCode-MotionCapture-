import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords, draw_status_box

class PushUpDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None # "up" or "down"
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        h, w, _ = image.shape

        shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value

        # Check visibility
        if not check_visibility(landmarks, [shoulder_idx, elbow_idx, wrist_idx]):
            return image

        l_shoulder = get_landmark_coords(landmarks, shoulder_idx, w, h)
        l_elbow = get_landmark_coords(landmarks, elbow_idx, w, h)
        l_wrist = get_landmark_coords(landmarks, wrist_idx, w, h)

        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        cv2.putText(image, str(int(angle)),
                           tuple(np.array(l_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Logic: Count on return to UP (Extension)
        # UP: > 160
        # DOWN: < 90

        if angle > 160:
            if self.stage == "down":
                self.counter += 1
                print(f"Push-up count: {self.counter}")
            self.stage = "up"

        if angle < 90:
            self.stage = "down"

        draw_status_box(image, "PUSH-UPS", self.counter)

        return image
