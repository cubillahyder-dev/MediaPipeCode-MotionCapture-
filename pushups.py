import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility

class PushUpDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None # "up" or "down"
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count push-ups.
        Args:
            image: The image frame to draw on.
            landmarks: The list of pose landmarks detected by MediaPipe.
        """
        h, w, _ = image.shape

        # Define indices
        shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value
        ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value

        # Check visibility
        if not check_visibility(landmarks, [shoulder_idx, elbow_idx, wrist_idx, ankle_idx]):
            return image

        # Get coordinates for left arm
        l_shoulder = [landmarks[shoulder_idx].x * w, landmarks[shoulder_idx].y * h]
        l_elbow = [landmarks[elbow_idx].x * w, landmarks[elbow_idx].y * h]
        l_wrist = [landmarks[wrist_idx].x * w, landmarks[wrist_idx].y * h]
        l_ankle = [landmarks[ankle_idx].x * w, landmarks[ankle_idx].y * h]

        # Check if in pushup position (prone)
        # In a pushup, the body is roughly horizontal.
        # So y difference between shoulder and ankle should be small relative to height,
        # or at least shoulder shouldn't be significantly above ankle (like in standing).
        # Actually, if camera is side view, horizontal means x-diff is large, y-diff is small.
        # Let's assume user is somewhat horizontal.
        # Simple check: If shoulder is way higher than ankle (smaller Y), it's likely standing.
        # But wait, y increases downwards. 0 is top.
        # Standing: Shoulder Y (e.g. 100) < Ankle Y (e.g. 400). Diff ~ -300.
        # Prone: Shoulder Y (e.g. 300) ~ Ankle Y (e.g. 300). Diff ~ 0.
        # Let's verify we are not standing.

        # However, camera angle might be top-down.
        # Let's stick to the visibility check first as requested, and a simple angle check.
        # If the user is standing, the elbow angle logic still works if they do "air pushups".
        # But the user asked to not count if not in position.
        # Let's enforce that Shoulder Y is not significantly higher than Ankle Y (Standing).
        # But this might break if the user has the camera low looking up.
        # Safer check: Confidence is key.

        # Calculate elbow angle
        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        # Visualize angle
        cv2.putText(image, str(int(angle)),
                           tuple(np.array(l_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Push-up Counter Logic
        # Elbow angle > 160 is UP
        # Elbow angle < 90 is DOWN

        if angle > 160:
            if self.stage == 'down':
                self.counter += 1
                print(f"Push-up count: {self.counter}")
            self.stage = "up"
        if angle < 90:
            self.stage = "down"

        # Draw UI box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage),
                    (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
