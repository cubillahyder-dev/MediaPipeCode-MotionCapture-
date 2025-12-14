import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility

class PunchDetector:
    def __init__(self):
        self.counter = 0
        self.stage_left = None
        self.stage_right = None
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count punches.
        Counts when an arm goes from flexed (guard) to extended (punch).
        """
        h, w, _ = image.shape

        # Indices
        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        l_elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        l_wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value

        r_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        r_elbow_idx = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
        r_wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

        # Visibility Check (Check individual arms independently or both? Let's check independently to allow single arm punches even if other is hidden)
        # Actually, for simplicity, let's try to track whatever is visible.

        l_visible = check_visibility(landmarks, [l_shoulder_idx, l_elbow_idx, l_wrist_idx])
        r_visible = check_visibility(landmarks, [r_shoulder_idx, r_elbow_idx, r_wrist_idx])

        # LEFT ARM
        if l_visible:
            l_shoulder = [landmarks[l_shoulder_idx].x * w, landmarks[l_shoulder_idx].y * h]
            l_elbow = [landmarks[l_elbow_idx].x * w, landmarks[l_elbow_idx].y * h]
            l_wrist = [landmarks[l_wrist_idx].x * w, landmarks[l_wrist_idx].y * h]

            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            cv2.putText(image, str(int(l_angle)),
                           tuple(np.array(l_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Logic
            # Guard/Flexed: < 90 (or < 100)
            # Punch/Extended: > 160

            if l_angle < 100:
                self.stage_left = "guard"
            if l_angle > 160 and self.stage_left == "guard":
                self.stage_left = "punch"
                self.counter += 1
                print(f"Punch count: {self.counter} (Left)")

        # RIGHT ARM
        if r_visible:
            r_shoulder = [landmarks[r_shoulder_idx].x * w, landmarks[r_shoulder_idx].y * h]
            r_elbow = [landmarks[r_elbow_idx].x * w, landmarks[r_elbow_idx].y * h]
            r_wrist = [landmarks[r_wrist_idx].x * w, landmarks[r_wrist_idx].y * h]

            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            cv2.putText(image, str(int(r_angle)),
                           tuple(np.array(r_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if r_angle < 100:
                self.stage_right = "guard"
            if r_angle > 160 and self.stage_right == "guard":
                self.stage_right = "punch"
                self.counter += 1
                print(f"Punch count: {self.counter} (Right)")

        # UI
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        cv2.putText(image, 'PUNCHES', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
