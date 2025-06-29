import cv2 as cv
import numpy as np

def prepare(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    moments = cv.moments(gray)
    hu = cv.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

def compare(hu1, hu2):
    return np.linalg.norm(hu1 - hu2)