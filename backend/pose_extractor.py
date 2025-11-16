# backend/pose_extractor.py
import mediapipe as mp
import numpy as np
import cv2
from backend.angles import calculate_angle

mp_pose = mp.solutions.pose

class PoseExtractor:
    def __init__(self):
        self.pose = mp_pose.Pose()

    def extract_angles(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        hip = calculate_angle(lm[23], lm[25], lm[27])
        knee = calculate_angle(lm[25], lm[27], lm[31])
        shoulder = calculate_angle(lm[11], lm[13], lm[15])

        return np.array([hip, knee, shoulder])
