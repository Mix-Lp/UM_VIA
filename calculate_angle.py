#!/usr/bin/env python

#This script is based on the one found in /code/util/medidor.py

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText

points = deque(maxlen=2)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

focal_length = 732 #pixels
camera_width = 800
camera_height = 600

for key, frame in autoStream():
    for p in points:
        cv.circle(frame, p,3,(0,0,255),-1)
    if len(points) == 2:
        cv.line(frame, points[0],points[1],(0,0,255))
        c = np.mean(points, axis=0).astype(int)
        #Lo calculamos de manera aproximada, asumiendo que el eje pasa por el centro de la matriz de pixels (en este caso
        # asumimos que el centro estará en el punto (400,300))
        u = [(points[0])[0] - camera_width/2, (points[0])[1] - camera_height/2, focal_length]
        v = [(points[1])[0] - camera_width/2, (points[1])[1] - camera_height/2, focal_length]
        a = np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
        a = np.rad2deg(a)
        putText(frame,f'{a:.5f} deg',c)

    cv.imshow('webcam',frame)
