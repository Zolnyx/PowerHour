import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

EXERCISES = [
    'squats',
    'planks',
]

# Angle between two vectors
def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    if mod_v1 == 0 or mod_v2 == 0:
        return 0
    cos_theta = np.clip(dot / (mod_v1 * mod_v2), -1.0, 1.0)
    return math.acos(cos_theta)

# Length of a vector
def get_length(v):
    return np.linalg.norm(v)

# Returns pose parameters for squats or other exercises
def get_params(results, exercise='squats', all=False):
    if results.pose_landmarks is None:
        if exercise == 'squats':
            return np.zeros((5,))
        else:
            return np.array([0, 0])

    # Required landmarks
    required = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_EYE", "RIGHT_EYE",
        "MOUTH_LEFT", "MOUTH_RIGHT"
    ]

    # Map from name to mediapipe enum
    name_map = {name: getattr(mp_pose.PoseLandmark, name) for name in required}

    # Get all points
    points = {}
    for name, landmark_enum in name_map.items():
        try:
            lm = results.pose_landmarks.landmark[landmark_enum]
            points[name] = np.array([lm.x, lm.y, lm.z])
        except:
            return np.zeros((5,))  # Skip frame if anything is missing

    # Midpoints
    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    z_eyes = (points["RIGHT_EYE"][2] + points["LEFT_EYE"][2]) / 2
    z_mouth = (points["MOUTH_LEFT"][2] + points["MOUTH_RIGHT"][2]) / 2

    theta_neck = get_angle(np.array([0, 0, -1]), points["NOSE"] - points["MID_HIP"])
    theta_s1 = get_angle(points["LEFT_ELBOW"] - points["LEFT_SHOULDER"], points["LEFT_HIP"] - points["LEFT_SHOULDER"])
    theta_s2 = get_angle(points["RIGHT_ELBOW"] - points["RIGHT_SHOULDER"], points["RIGHT_HIP"] - points["RIGHT_SHOULDER"])
    theta_s = (theta_s1 + theta_s2) / 2

    theta_k1 = get_angle(points["RIGHT_HIP"] - points["RIGHT_KNEE"], points["RIGHT_ANKLE"] - points["RIGHT_KNEE"])
    theta_k2 = get_angle(points["LEFT_HIP"] - points["LEFT_KNEE"], points["LEFT_ANKLE"] - points["LEFT_KNEE"])
    theta_k = (theta_k1 + theta_k2) / 2

    theta_h1 = get_angle(points["RIGHT_KNEE"] - points["RIGHT_HIP"], points["RIGHT_SHOULDER"] - points["RIGHT_HIP"])
    theta_h2 = get_angle(points["LEFT_KNEE"] - points["LEFT_HIP"], points["LEFT_SHOULDER"] - points["LEFT_HIP"])
    theta_h = (theta_h1 + theta_h2) / 2

    torso_length = get_length(points['MID_SHOULDER'] - points['MID_HIP'])
    left_thigh_length = get_length(points['LEFT_KNEE'] - points['LEFT_HIP'])
    right_thigh_length = get_length(points['RIGHT_KNEE'] - points['RIGHT_HIP'])
    left_tibula_length = get_length(points['LEFT_KNEE'] - points['LEFT_HEEL'])
    right_tibula_length = get_length(points['RIGHT_KNEE'] - points['RIGHT_HEEL'])
    thigh_length = (left_thigh_length + right_thigh_length) / 2
    tibula_length = (left_tibula_length + right_tibula_length) / 2
    length_normalization_factor = (1 / (tibula_length + 1e-6))**0.5  # Avoid div by zero

    z1 = (points["RIGHT_ANKLE"][2] + points["RIGHT_HEEL"][2]) / 2 - points["RIGHT_FOOT_INDEX"][2]
    z2 = (points["LEFT_ANKLE"][2] + points["LEFT_HEEL"][2]) / 2 - points["LEFT_FOOT_INDEX"][2]
    z = ((z1 + z2) / 2) * length_normalization_factor

    left_foot_y = (points["LEFT_ANKLE"][1] + points["LEFT_HEEL"][1] + points["LEFT_FOOT_INDEX"][1]) / 3
    right_foot_y = (points["RIGHT_ANKLE"][1] + points["RIGHT_HEEL"][1] + points["RIGHT_FOOT_INDEX"][1]) / 3
    left_ky = points["LEFT_KNEE"][1] - left_foot_y
    right_ky = points["RIGHT_KNEE"][1] - right_foot_y
    ky = ((left_ky + right_ky) / 2) * length_normalization_factor

    if exercise == 'squats':
        params = np.array([theta_neck, theta_k, theta_h, z, ky])

    if all:
        params = np.array([[x, y, z] for pos, (x, y, z) in points.items()]) * length_normalization_factor

    return np.round(params, 2)
