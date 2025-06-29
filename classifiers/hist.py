import cv2 as cv

def prepare(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    return cv.normalize(hist, hist).flatten()

def compare(h1, h2):
    return 1 - cv.compareHist(h1, h2, cv.HISTCMP_CORREL)  # menor = más parecido
