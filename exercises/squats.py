import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords, draw_status_box, draw_info_box, is_pose_standing

class SquatDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose
        self.is_ready = False

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Check Posture - Squats are tricky because "in position" means standing, but during rep you are not "standing"
        # However, is_pose_standing checks y-coordinates relative order (Shoulder < Hip < Knee).
        # Even in a deep squat, your shoulders are above hips, and hips above knees (mostly, unless deep deep squat).
        # Actually in deep squat, hips can be below knees.
        # So strict `is_pose_standing` might fail at bottom of squat.
        # But we only need to validat "Ready" state at the start or top.

        # Let's say: If we are not recording a rep, we demand standing.
        # If we are in the middle of a rep, we relax the check?
        # Or simpler: The user demanded "make sure user is in position first".
        # This implies checking before STARTING the workout or the first rep.

        # Let's check standing if self.stage is None or 'up'.

        check_posture = (self.stage is None) or (self.stage == 'up')

        if check_posture:
            if not is_pose_standing(landmarks):
                draw_info_box(image, "Please Stand Up", color=(0,0,255))
                self.is_ready = False
                # If we are not ready, do we return? If we return, we stop processing.
                # But we might want to continue processing to see if they stand up.
                # But we shouldn't count.
                draw_status_box(image, "SQUATS", self.counter)
                return image
            else:
                self.is_ready = True
                draw_info_box(image, "Ready", color=(0,255,0))

        if not self.is_ready:
            # If not ready (and not in the middle of a rep), just return
             draw_status_box(image, "SQUATS", self.counter)
             return image

        hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        knee_idx = self.mp_pose.PoseLandmark.LEFT_KNEE.value
        ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        if not check_visibility(landmarks, [hip_idx, knee_idx, ankle_idx]):
            return image

        hip = get_landmark_coords(landmarks, hip_idx, w, h)
        knee = get_landmark_coords(landmarks, knee_idx, w, h)
        ankle = get_landmark_coords(landmarks, ankle_idx, w, h)

        angle = calculate_angle(hip, knee, ankle)

        cv2.putText(image, str(int(angle)),
                           tuple(np.array(knee).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Squat logic
        # Standing: > 160
        # Squat: < 100

        if angle > 160:
            if self.stage == "down":
                self.counter += 1
                print(f"Squat count: {self.counter}")
            self.stage = "up"

        if angle < 100:
            self.stage = "down"

        draw_status_box(image, "SQUATS", self.counter)

        return image
