#!/usr/bin/env python

#Script borrowed from umucv/code/DL/UNET
import numpy as np
import cv2 as cv
import sys

filename = sys.argv[1]
img = cv.imread(filename)

if img is None:
    print(f"[ERROR] No se pudo leer la imagen: {filename}")
    sys.exit(1)

print(f"[INFO] Editando máscara de: {filename}")
print("[INFO] Instrucciones:")
print("  - Clic y arrastra para dibujar")
print("  - Tecla 'x' para deshacer último punto")
print("  - Tecla 'q' para salir y guardar")
print("  - Tecla 'ESC' para salir SIN guardar")

status = [False]
points = []

def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        status[0] = True
    if event == cv.EVENT_LBUTTONUP:
        status[0] = False
    if status[0] and event == 0:
        points.append((x,y))

cv.namedWindow("mask")
cv.setMouseCallback("mask", manejador)

while True:
    key = cv.waitKey(100) & 0xFF
    if key == ord('x') and points:
        del points[-1]
    elif key == ord('q'):
        break
    elif key == 27:
        points.clear()
        break

    preview = img.copy()
    for (x, y) in points:
        cv.circle(preview, (x, y), 10, color=(0, 0, 255), thickness=-1)

    cv.imshow("mask", preview)

cv.destroyAllWindows()

# Solo guarda si hay puntos
if points:
    result = np.zeros_like(img)
    for (x, y) in points:
        cv.circle(result, (x, y), 10, color=(255, 255, 255), thickness=-1)
    outname = "mask_" + filename
    cv.imwrite(outname, result)
    print(f"[INFO] Máscara guardada como {outname}")
else:
    print("[INFO] No se marcó ninguna zona. No se guardó nada.")
