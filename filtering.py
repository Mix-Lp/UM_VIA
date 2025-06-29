#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import Help, ROI

help = Help(
    """
    BLUR FILTERS

    0: do nothing
    1: box
    2: Gaussian
    3: median
    4: bilateral
    5: min
    6: max
    7: box (integral)

    c: color/monochrome
    r: only roi
    i: invert on/off
    g: toggle Gaussian separability

    SPC: pause

    h: show/hide help
    """
)

color = False
invert = False
current_filter = 0
paused = False
roi_mode = False
gaussian_separability = False

# Parámetros para los filtros
cv.namedWindow('result')
cv.createTrackbar('Box Size', 'result', 15, 100, lambda v: None)
cv.createTrackbar('Gaussian Sigma', 'result', 3, 100, lambda v: None)
cv.createTrackbar('Gaussian Cascading', 'result', 1, 10, lambda v: None)
cv.createTrackbar('Min/Max Size', 'result', 15, 100, lambda v: None)

# Inicializar ROI
region = ROI("result")

# Filtro con imagen integral
def box_filter_integral(img, ksize):
    if ksize % 2 == 0:
        ksize += 1
    r = ksize // 2

    # Si es color, aplicar por canal
    if len(img.shape) == 3:
        channels = cv.split(img)
        filtered = [box_filter_integral(c, ksize) for c in channels]
        return cv.merge(filtered)

    # Convertir a float32 para evitar overflow
    img = img.astype(np.float32)
    h, w = img.shape
    integral = cv.integral(img)[1:, 1:]

    result = np.zeros_like(img)

    for y in range(h):
        y1 = max(y - r, 0)
        y2 = min(y + r, h - 1)
        for x in range(w):
            x1 = max(x - r, 0)
            x2 = min(x + r, w - 1)

            A = integral[y1 - 1, x1 - 1] if y1 > 0 and x1 > 0 else 0
            B = integral[y1 - 1, x2] if y1 > 0 else 0
            C = integral[y2, x1 - 1] if x1 > 0 else 0
            D = integral[y2, x2]

            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            result[y, x] = (D - B - C + A) / area

    return result.astype(img.dtype)


for key, frame in autoStream():
    help.show_if(key, ord('h'))

    if key == ord('0'):
        current_filter = 0
    if key in [ord(str(i)) for i in range(1, 8)]:
        current_filter = int(chr(key))
    if key == ord('c'):
        color = not color
    if key == ord('i'):
        invert = not invert
    if key == ord('r'):
        roi_mode = not roi_mode
    if key == ord('g'):
        gaussian_separability = not gaussian_separability
    if key == ord(' '):
        paused = not paused

    if paused:
        continue

    # Preprocesado: convertir a escala de grises si no se solicita color
    if not color:
        result = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        result = frame

    # Invertir imagen si se solicita
    if invert:
        result = 255 - result

    # Obtener parámetros de trackbars
    box_ksize = cv.getTrackbarPos('Box Size', 'result') or 1
    gaussian_sigma = cv.getTrackbarPos('Gaussian Sigma', 'result')
    gaussian_cascading = cv.getTrackbarPos('Gaussian Cascading', 'result') or 1
    min_max_ksize = cv.getTrackbarPos('Min/Max Size', 'result') or 1

    # Aplicación de los filtros
    def apply_filter(img):
        if current_filter == 1:
            return cv.boxFilter(img, -1, (box_ksize, box_ksize))
        elif current_filter == 2:
            filtered = img.copy()
            for _ in range(gaussian_cascading):
                ksize = max(1, 2 * int(3 * gaussian_sigma) + 1)  # Asegurar que el tamaño sea impar y mayor que 0
                if gaussian_separability:
                    kernel = cv.getGaussianKernel(ksize, gaussian_sigma)
                    filtered = cv.sepFilter2D(filtered, -1, kernel, kernel)
                else:
                    filtered = cv.GaussianBlur(filtered, (ksize, ksize), gaussian_sigma)
            return filtered
        elif current_filter == 3:
            return cv.medianBlur(img, 15)
        elif current_filter == 4:
            return cv.bilateralFilter(img, 9, 75, 75)
        elif current_filter == 5:
            kernel = np.ones((min_max_ksize, min_max_ksize), np.uint8)
            return cv.erode(img, kernel)
        elif current_filter == 6:
            kernel = np.ones((min_max_ksize, min_max_ksize), np.uint8)
            return cv.dilate(img, kernel)
        elif current_filter == 7:
            return box_filter_integral(img, box_ksize)
        return img

    if roi_mode and region.roi:
        x1, y1, x2, y2 = region.roi
        roi = result[y1:y2 + 1, x1:x2 + 1]
        filtered_roi = apply_filter(roi)
        result[y1:y2 + 1, x1:x2 + 1] = filtered_roi
        cv.rectangle(result, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
    else:
        result = apply_filter(result)

    # Mostrar la imagen procesada
    cv.imshow('result', result)
