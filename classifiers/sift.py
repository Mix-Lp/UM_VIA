#Script based on /code/SIFT/sift.py

import cv2 as cv

sift = cv.SIFT_create(nfeatures=1000)
matcher = cv.BFMatcher()

model_keypoints = {}
model_descriptors = {}
model_images = {}
show_matches = False

def prepare(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kps, des = sift.detectAndCompute(gray, None)
    return (kps, des)

def compare(query_feat, model_feat):
    qkps, qdes = query_feat
    mkps, mdes = model_feat

    if qdes is None or mdes is None:
        return float("inf")

    matches = matcher.knnMatch(qdes, mdes, k=2)
    good = []
    for m in matches:
        if len(m) >= 2:
            best, second = m
            if best.distance < 0.55 * second.distance:
                good.append(best)

    return 1 / (len(good) + 1e-5)

#Draw thumbnail to better distinguish model
def draw_best_match(frame, best_name, model_path):
    thumb = cv.imread(model_path)
    if thumb is not None:
        thumb = cv.resize(thumb, (100, 100))
        h, w = frame.shape[:2]
        frame[0:100, w - 100:w] = thumb

def match_and_draw(frame, query_feat, model_feat):
    global show_matches

    if not show_matches:
        return frame

    qkps, qdes = query_feat
    mkps, mdes = model_feat

    if qdes is None or mdes is None:
        return frame

    matches = matcher.knnMatch(qdes, mdes, k=2)
    good = []
    for m in matches:
        if len(m) >= 2:
            best, second = m
            if best.distance < 0.55 * second.distance:
                good.append(best)

    imgm = cv.drawMatches(frame, qkps, model_images.get('current', frame), mkps, good,
                          flags=0,
                          matchColor=(128,255,128),
                          singlePointColor=(128,128,128),
                          outImg=None)

    return imgm

def set_visualization(flag):
    global show_matches
    show_matches = flag
