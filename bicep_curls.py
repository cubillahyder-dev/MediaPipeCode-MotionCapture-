import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle, check_visibility

class BicepCurlDetector:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose

    def process(self, image, landmarks):
        """
        Processes the image and landmarks to count bicep curls.
        """
        h, w, _ = image.shape

        # Use left arm
        shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value

        # Check visibility
        if not check_visibility(landmarks, [shoulder_idx, elbow_idx, wrist_idx]):
            return image

        # Get coordinates
        l_shoulder = [landmarks[shoulder_idx].x * w, landmarks[shoulder_idx].y * h]
        l_elbow = [landmarks[elbow_idx].x * w, landmarks[elbow_idx].y * h]
        l_wrist = [landmarks[wrist_idx].x * w, landmarks[wrist_idx].y * h]

        # Calculate elbow angle
        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        # Visualize angle
        cv2.putText(image, str(int(angle)),
                           tuple(np.array(l_elbow).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Curl Logic
        # Down (Extended): > 160
        # Up (Flexed): < 30

        if angle > 160:
            self.stage = "down"
        if angle < 30 and self.stage == 'down':
            self.stage = "up"
            self.counter += 1
            print(f"Curl count: {self.counter}")

        # UI
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage),
                    (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        return image
