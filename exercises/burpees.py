import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords, draw_status_box

class BurpeeDetector:
    def __init__(self):
        self.counter = 0
        self.stage = "standing" # standing -> squat -> plank -> pushup (opt) -> squat -> standing
        self.mp_pose = mp.solutions.pose
        self.history = []

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Burpee is complex.
        # 1. Standing (Vertical, hands down)
        # 2. Squat (Hips low)
        # 3. Plank (Horizontal)
        # 4. Squat
        # 5. Jump/Stand

        # Key metrics:
        # Hip Y coord relative to Shoulder Y (Verticality)
        # Hip Angle (Squat)
        # Shoulder-Hip-Ankle Angle (Plank)

        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        l_hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        l_knee_idx = self.mp_pose.PoseLandmark.LEFT_KNEE.value
        l_ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        if not check_visibility(landmarks, [l_shoulder_idx, l_hip_idx, l_knee_idx, l_ankle_idx]):
            return image

        l_shoulder = get_landmark_coords(landmarks, l_shoulder_idx, w, h)
        l_hip = get_landmark_coords(landmarks, l_hip_idx, w, h)
        l_knee = get_landmark_coords(landmarks, l_knee_idx, w, h)
        l_ankle = get_landmark_coords(landmarks, l_ankle_idx, w, h)

        # Angles
        hip_knee_ankle_angle = calculate_angle(l_hip, l_knee, l_ankle)
        shoulder_hip_knee_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        shoulder_hip_ankle_angle = calculate_angle(l_shoulder, l_hip, l_ankle) # For plank

        cv2.putText(image, f"Stage: {self.stage}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # State Transitions
        if self.stage == "standing":
            # Check for squat or plank (if fast)
            # Squat: Hip angle < 100
            if hip_knee_ankle_angle < 100:
                self.stage = "squat_down"

        elif self.stage == "squat_down":
            # Check for plank
            # Plank: Body straight ~180
            if shoulder_hip_ankle_angle > 160:
                self.stage = "plank"
            # Or go back to standing if failed
            elif hip_knee_ankle_angle > 160:
                self.stage = "standing"

        elif self.stage == "plank":
            # Check for return to squat
            if hip_knee_ankle_angle < 100:
                self.stage = "squat_up"

        elif self.stage == "squat_up":
            # Check for standing/jump
            if hip_knee_ankle_angle > 160:
                self.stage = "standing"
                self.counter += 1

        draw_status_box(image, "BURPEES", self.counter)

        return image
