#Script based on /code/DL/mp_hands/hands.py
import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.spatial import procrustes

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

visualize = False

def set_visualization(enabled=True):
    global visualize
    visualize = enabled

def extract_landmarks(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    if visualize:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    landmarks = results.multi_hand_landmarks[0].landmark
    return np.array([[lm.x, lm.y] for lm in landmarks])

def normalize_landmarks(landmarks):
    if landmarks is None:
        return None
    return (landmarks - landmarks.mean(axis=0)) / landmarks.std(axis=0)

def prepare(img):
    landmarks = extract_landmarks(img)
    return normalize_landmarks(landmarks)

def compare(l1, l2):
    if l1 is None or l2 is None:
        return float('inf')
    try:
        mtx1, mtx2, disparity = procrustes(l1, l2)
        return disparity
    except Exception:
        return float('inf')



"""
#!/usr/bin/env python

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import cv2 as cv
from umucv.stream import autoStream

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    for key,frame in autoStream():
        frame = cv.flip(frame,1);
        h,w = frame.shape[:2]

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        cv.imshow('MediaPipe Hands', frame)

"""