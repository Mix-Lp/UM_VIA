#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from collections import deque
from umucv.util import putText
import sys

if len(sys.argv) < 2:
    print("Uso: ./show_pixels.py imagen.jpg")
    sys.exit(1)

img = cv.imread(sys.argv[1])
if img is None:
    print("Error: no se pudo cargar la imagen")
    sys.exit(1)

points = deque(maxlen=2)

def click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Click en: ({x}, {y})")

cv.namedWindow("imagen")
cv.setMouseCallback("imagen", click)

while True:
    display = img.copy()
    for p in points:
        cv.circle(display, p, 5, (0, 0, 255), -1)
        putText(display, f'({p[0]}, {p[1]})', (p[0] + 10, p[1] - 10))

    if len(points) == 2:
        cv.line(display, points[0], points[1], (0, 0, 255), 2)

    cv.imshow("imagen", display)
    if cv.waitKey(10) == ord('q'):
        break

cv.destroyAllWindows()
