import cv2 as cv
import numpy as np

orb = cv.ORB_create()

def prepare(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kps, des = orb.detectAndCompute(gray, None)
    return des if des is not None else np.array([])

def compare(desc1, desc2):
    if len(desc1) == 0 or len(desc2) == 0:
        return float("inf")
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return np.mean([m.distance for m in matches]) if matches else float("inf")
