import numpy as np

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
