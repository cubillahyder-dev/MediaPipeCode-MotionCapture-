import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility, get_landmark_coords, draw_status_box, draw_info_box, is_pose_prone

class MountainClimberDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None # 'left' or 'right' leg forward
        self.mp_pose = mp.solutions.pose
        self.prev_leg = None
        self.is_ready = False

    def process(self, image, landmarks):
        h, w, _ = image.shape

        # Check Posture - Prone
        if not is_pose_prone(landmarks):
            draw_info_box(image, "Please Lie Down", color=(0,0,255))
            self.is_ready = False
            draw_status_box(image, "MTN CLIMBERS", self.counter)
            return image
        else:
            self.is_ready = True
            draw_info_box(image, "Ready", color=(0,255,0))

        # We need hips and knees to detect leg movement
        # Also maybe shoulders to ensure plank position?
        # For simplicity, let's just count knee to chest movements.

        l_hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
        l_knee_idx = self.mp_pose.PoseLandmark.LEFT_KNEE.value
        l_ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        r_hip_idx = self.mp_pose.PoseLandmark.RIGHT_HIP.value
        r_knee_idx = self.mp_pose.PoseLandmark.RIGHT_KNEE.value
        r_ankle_idx = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value

        # Visibility check
        if not check_visibility(landmarks, [l_hip_idx, l_knee_idx, r_hip_idx, r_knee_idx]):
            return image

        l_hip = get_landmark_coords(landmarks, l_hip_idx, w, h)
        l_knee = get_landmark_coords(landmarks, l_knee_idx, w, h)
        l_ankle = get_landmark_coords(landmarks, l_ankle_idx, w, h)

        r_hip = get_landmark_coords(landmarks, r_hip_idx, w, h)
        r_knee = get_landmark_coords(landmarks, r_knee_idx, w, h)
        r_ankle = get_landmark_coords(landmarks, r_ankle_idx, w, h)

        # Calculate angles of hip flexion?
        # Or simpler: Distance between knee and hip?
        # Mountain climber: Knee comes close to chest (or hip).
        # We can look at the angle at the hip: Shoulder-Hip-Knee.

        # Let's get shoulders
        l_shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value

        if not check_visibility(landmarks, [l_shoulder_idx, r_shoulder_idx]):
             # If shoulders not visible, assume horizontal alignment or skip
             # Assume hips are visible.
             pass

        l_shoulder = get_landmark_coords(landmarks, l_shoulder_idx, w, h)
        r_shoulder = get_landmark_coords(landmarks, r_shoulder_idx, w, h)

        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

        # Visualize
        cv2.putText(image, f"L:{int(l_hip_angle)}", tuple(np.array(l_hip).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"R:{int(r_hip_angle)}", tuple(np.array(r_hip).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Logic:
        # Standing straight: Hip angle ~ 180.
        # Knee to chest: Hip angle < 90 (or less).
        # We count when one knee goes < threshold (e.g. 100) then returns > threshold.

        # State machine is tricky with two legs.
        # Let's count total reps (each leg = 1 rep)

        threshold_in = 100
        threshold_out = 150

        # Left leg state
        if not hasattr(self, 'left_stage'): self.left_stage = 'out'
        if not hasattr(self, 'right_stage'): self.right_stage = 'out'

        if l_hip_angle < threshold_in:
            self.left_stage = 'in'
        if l_hip_angle > threshold_out and self.left_stage == 'in':
            self.left_stage = 'out'
            self.counter += 1

        if r_hip_angle < threshold_in:
            self.right_stage = 'in'
        if r_hip_angle > threshold_out and self.right_stage == 'in':
            self.right_stage = 'out'
            self.counter += 1

        draw_status_box(image, "MTN CLIMBERS", self.counter)

        return image
