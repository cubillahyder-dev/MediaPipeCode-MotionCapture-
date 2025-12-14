import cv2
import numpy as np
import mediapipe as mp
from pose_utils import calculate_angle

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

        # Get coordinates for left arm
        l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate elbow angle
        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        # Visualize angle
        h, w, _ = image.shape
        cv2.putText(image, str(int(angle)),
                           tuple(np.multiply(l_elbow, [w, h]).astype(int)),
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
