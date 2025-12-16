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

def draw_info_box(image, text, color=(0,0,0), text_color=(255,255,255)):
    """
    Draws a notifier box on the top-left corner of the image.
    """
    # Define box dimensions
    box_width = 300
    box_height = 50

    start_point = (0, 0)
    end_point = (box_width, box_height)

    # Draw background with some transparency? For now just solid.
    cv2.rectangle(image, start_point, end_point, color, -1)

    cv2.putText(image, text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

def is_pose_standing(landmarks):
    """
    Simple heuristic to check if the person is standing.
    Checks if Shoulders are above Hips, and Hips above Knees (if visible).
    """
    # We use indices: 11 (Left Shoulder), 23 (Left Hip), 25 (Left Knee)
    #               12 (Right Shoulder), 24 (Right Hip), 26 (Right Knee)

    # Note: Y coordinate increases downwards (0 is top).
    # So Shoulder Y < Hip Y < Knee Y.

    # Check Left side
    l_shoulder_y = landmarks[11].y
    l_hip_y = landmarks[23].y
    l_knee_y = landmarks[25].y

    # Check Right side
    r_shoulder_y = landmarks[12].y
    r_hip_y = landmarks[24].y
    r_knee_y = landmarks[26].y

    # Check visibility before strictly enforcing
    l_visible = check_visibility(landmarks, [11, 23, 25])
    r_visible = check_visibility(landmarks, [12, 24, 26])

    # If neither side is fully visible, we might have issues.
    # Let's try to use hips and shoulders at least.

    standing_l = (l_shoulder_y < l_hip_y)
    standing_r = (r_shoulder_y < r_hip_y)

    if l_visible:
        standing_l = standing_l and (l_hip_y < l_knee_y)

    if r_visible:
        standing_r = standing_r and (r_hip_y < r_knee_y)

    return standing_l or standing_r

def is_pose_prone(landmarks):
    """
    Simple heuristic to check if the person is in prone position (pushup/plank).
    Checks if the body is roughly horizontal.
    Shoulder Y approx equal to Hip Y.
    """
    l_shoulder_y = landmarks[11].y
    l_hip_y = landmarks[23].y

    r_shoulder_y = landmarks[12].y
    r_hip_y = landmarks[24].y

    # Tolerance for horizontal alignment (e.g. 0.2 of image height)
    tolerance = 0.2

    horizontal_l = abs(l_shoulder_y - l_hip_y) < tolerance
    horizontal_r = abs(r_shoulder_y - r_hip_y) < tolerance

    return horizontal_l or horizontal_r
