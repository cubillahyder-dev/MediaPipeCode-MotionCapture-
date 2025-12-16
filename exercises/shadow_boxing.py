import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords, draw_status_box

class ShadowBoxingDetector:
    def __init__(self):
        self.counter = 0
        self.stage_left = None
        self.stage_right = None
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Indices
        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        l_elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        l_wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value

        r_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        r_elbow_idx = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
        r_wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

        l_visible = check_visibility(landmarks, [l_shoulder_idx, l_elbow_idx, l_wrist_idx])
        r_visible = check_visibility(landmarks, [r_shoulder_idx, r_elbow_idx, r_wrist_idx])

        # LEFT ARM
        if l_visible:
            l_shoulder = get_landmark_coords(landmarks, l_shoulder_idx, w, h)
            l_elbow = get_landmark_coords(landmarks, l_elbow_idx, w, h)
            l_wrist = get_landmark_coords(landmarks, l_wrist_idx, w, h)

            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # Visualize
            cv2.putText(image, str(int(l_angle)),
                           tuple(np.array(l_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if l_angle < 90:
                self.stage_left = "guard"
            if l_angle > 160 and self.stage_left == "guard":
                self.stage_left = "punch"
                self.counter += 1

        # RIGHT ARM
        if r_visible:
            r_shoulder = get_landmark_coords(landmarks, r_shoulder_idx, w, h)
            r_elbow = get_landmark_coords(landmarks, r_elbow_idx, w, h)
            r_wrist = get_landmark_coords(landmarks, r_wrist_idx, w, h)

            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            # Visualize
            cv2.putText(image, str(int(r_angle)),
                           tuple(np.array(r_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if r_angle < 90:
                self.stage_right = "guard"
            if r_angle > 160 and self.stage_right == "guard":
                self.stage_right = "punch"
                self.counter += 1

        # Draw UI
        draw_status_box(image, "PUNCHES", self.counter)

        return image
