#!/usr/bin/env python

import cv2 as cv
import numpy as np
import time
from umucv.util import putText


def manual_convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    result = np.zeros_like(image, dtype=np.float32)

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Convolución manual
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result


image = cv.imread('image.png', cv.IMREAD_GRAYSCALE)

# Kernel de internet
kernel = np.array([[1*2, 2*2, 1*2], [2*2, 4*2, 2*2], [1*2, 2*2, 1*2]], dtype=np.float32) / 16

# Medidas de tiempo

start_manual = time.time()
manual_result = manual_convolution(image, kernel)
end_manual = time.time()

start_cv = time.time()
opencv_result = cv.filter2D(image, -1, kernel)
end_cv = time.time()


manual_time = (end_manual - start_manual) * 1000 # Milisegundos
opencv_time = (end_cv - start_cv) * 1000

print(f'Tiempo manual: {manual_time:.2f} ms')
print(f'Tiempo OpenCV: {opencv_time:.2f} ms')

cv.imshow('Original', image)
putText(manual_result, f'Manual: {manual_time:.2f} ms')
cv.imshow('Manual Convolution', np.uint8(manual_result))
putText(opencv_result, f'OpenCV: {opencv_time:.2f} ms')
cv.imshow('OpenCV Convolution', opencv_result)
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()