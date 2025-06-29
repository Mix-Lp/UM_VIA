#!/usr/bin/env python

import cv2 as cv
import numpy as np
import sys
from collections import deque

if len(sys.argv) < 3:
    print("Uso: ./_rectify_plane.py imagen.jpg refpoints.txt")
    sys.exit(1)

# Cargar imagen
img = cv.imread(sys.argv[1])
if img is None:
    print(f"Error: no se pudo cargar la imagen {sys.argv[1]}")
    sys.exit(1)

# Cargar puntos de referencia desde el archivo
ref_file = sys.argv[2]
with open(ref_file) as f:
    lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]
    if len(lines) != 4:
        print("Error: el archivo de referencia debe contener exactamente 4 puntos")
        sys.exit(1)
    points_img = [tuple(map(float, l[:2])) for l in lines]
    points_real = [tuple(map(float, l[2:])) for l in lines]

# Convertir puntos a np.array
src = np.array(points_img, dtype=np.float32)
dst = np.array(points_real, dtype=np.float32)

# Escala de 100px por cm
SCALE = 100
dst_scaled = dst * SCALE

# Calcular homografia
H_mat, _ = cv.findHomography(src, dst_scaled)

# Proyectar bounding box de la imagen original para la imagen warpeada
h, w = img.shape[:2]
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
new_corners = cv.perspectiveTransform(corners, H_mat)

[xmin, ymin] = np.min(new_corners, axis=0).flatten()
[xmax, ymax] = np.max(new_corners, axis=0).flatten()

new_w = int(np.ceil(xmax - xmin))
new_h = int(np.ceil(ymax - ymin))

# Asegurar que la imagen warpeada no se sale
T = np.array([[1, 0, -xmin],
              [0, 1, -ymin],
              [0, 0, 1]])
full_H = T @ H_mat

rectified_img = cv.warpPerspective(img, full_H, (new_w, new_h), flags=cv.INTER_LINEAR)

cv.namedWindow("Imagen Original", cv.WINDOW_NORMAL)
cv.namedWindow("Imagen Rectificada", cv.WINDOW_NORMAL)
cv.resizeWindow("Imagen Rectificada", 1000, 800)
cv.resizeWindow("Imagen Original", 1000, 800)

clicks = deque(maxlen=2)

def medir_distancia(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

cv.setMouseCallback("Imagen Original", medir_distancia)

while True:
    orig_display = img.copy()
    for p in clicks:
        cv.circle(orig_display, p, 5, (0, 0, 255), -1)
    if len(clicks) == 2:
        cv.line(orig_display, clicks[0], clicks[1], (255, 0, 255), 2)
        p1 = np.array([[clicks[0]]], dtype=np.float32)
        p2 = np.array([[clicks[1]]], dtype=np.float32)
        p1_real = cv.perspectiveTransform(p1, H_mat)[0][0]
        p2_real = cv.perspectiveTransform(p2, H_mat)[0][0]
        dist = np.linalg.norm(p2_real - p1_real) / SCALE  # distancia original (dividido entre la escala, 100px por cm)
        c = tuple(np.mean(clicks, axis=0).astype(int))
        cv.putText(orig_display, f"{dist:.2f} cm", c, cv.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 3)

    cv.imshow("Imagen Original", orig_display)
    cv.imshow("Imagen Rectificada", rectified_img)
    if cv.waitKey(10) == ord('q'):
        break

cv.destroyAllWindows()
