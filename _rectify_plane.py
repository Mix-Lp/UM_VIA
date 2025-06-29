#!/usr/bin/env python

import cv2 as cv
import numpy as np
import sys
from collections import deque

if len(sys.argv) < 2:
    print("Uso: ./_rectify_plane.py imagen.jpg")
    sys.exit(1)

img = cv.imread(sys.argv[1])
if img is None:
    print(f"Error: no se pudo cargar la imagen {sys.argv[1]}")
    sys.exit(1)

# Asumimos dimensiones del DNI
REAL_WIDTH_CM = 8.6
REAL_HEIGHT_CM = 5.4

points = []
clone = img.copy()


def draw_points():
    disp = clone.copy()
    for i, (x, y) in enumerate(points):
        cv.circle(disp, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv.putText(disp, str(i + 1), (int(x) + 5, int(y) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    if len(points) == 4:
        cv.polylines(disp, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    return disp


def rectifica_full():
    src = np.array(points, dtype=np.float32)

    # Dimensiones marcadas
    width_pix = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    height_pix = np.linalg.norm(np.array(points[1]) - np.array(points[2]))

    pixel_per_cm_w = width_pix / REAL_WIDTH_CM
    pixel_per_cm_h = height_pix / REAL_HEIGHT_CM
    px_per_cm = (pixel_per_cm_w + pixel_per_cm_h) / 2  # media

    dst = np.array([[0, 0], [width_pix - 1, 0], [width_pix - 1, height_pix - 1], [0, height_pix - 1]], dtype=np.float32)
    H_mat = cv.getPerspectiveTransform(src, dst)

    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    new_corners = cv.perspectiveTransform(corners, H_mat)

    [xmin, ymin] = np.min(new_corners, axis=0).flatten()
    [xmax, ymax] = np.max(new_corners, axis=0).flatten()

    new_w = int(np.ceil(xmax - xmin))
    new_h = int(np.ceil(ymax - ymin))

    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0, 1]])
    full_H = T @ H_mat

    warped = cv.warpPerspective(img, full_H, (new_w, new_h))
    return warped, px_per_cm, full_H


def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))


cv.namedWindow("Selecciona 4 puntos")
cv.setMouseCallback("Selecciona 4 puntos", select_points)

while True:
    canvas = draw_points()
    cv.imshow("Selecciona 4 puntos", canvas)
    if len(points) == 4:
        break
    if cv.waitKey(10) == ord('q'):
        sys.exit(0)

cv.destroyWindow("Selecciona 4 puntos")

# Aplicamos la homografía y guardamos info para medición
rectified_img, px_per_cm, H_mat = rectifica_full()
clicks = deque(maxlen=2)


def medir_distancia(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        clicks.append((x, y))


cv.namedWindow("Imagen rectificada")
cv.setMouseCallback("Imagen rectificada", medir_distancia)

while True:
    display = rectified_img.copy()
    for p in clicks:
        cv.circle(display, p, 5, (0, 0, 255), -1)
    if len(clicks) == 2:
        cv.line(display, clicks[0], clicks[1], (255, 0, 255), 2)
        c = np.mean(clicks, axis=0).astype(int)
        pix_dist = np.linalg.norm(np.array(clicks[0]) - np.array(clicks[1]))
        cm_dist = pix_dist / px_per_cm
        cv.putText(display, f"{cm_dist:.2f} cm", tuple(c), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv.imshow("Imagen rectificada", display)
    if cv.waitKey(10) == ord('q'):
        break

cv.destroyAllWindows()
