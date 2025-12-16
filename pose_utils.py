import numpy as np
import cv2

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, and c.
    a, b, c are (x, y) coordinates.
    The angle is calculated at point b.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def check_visibility(landmarks, indices, threshold=0.5):
    """
    Checks if the specified landmarks have visibility above the threshold.
    Args:
        landmarks: List of landmarks.
        indices: List of landmark indices to check.
        threshold: Minimum visibility score.
    Returns:
        True if all specified landmarks are visible, False otherwise.
    """
    for index in indices:
        if landmarks[index].visibility < threshold:
            return False
    return True

def get_landmark_coords(landmarks, landmark_idx, image_width, image_height):
    """
    Helper to get (x, y) coordinates of a landmark.
    """
    return [landmarks[landmark_idx].x * image_width, landmarks[landmark_idx].y * image_height]

def draw_status_box(image, label, value, color=(245,117,16), text_color=(255,255,255)):
    """
    Draws a standardized status box on the top-right corner of the image.
    """
    h, w, _ = image.shape

    # Define box dimensions
    box_width = 225
    box_height = 73

    start_point = (w - box_width, 0)
    end_point = (w, box_height)

    cv2.rectangle(image, start_point, end_point, color, -1)

    cv2.putText(image, str(label), (w - box_width + 15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(value),
                (w - box_width + 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 2, cv2.LINE_AA)
