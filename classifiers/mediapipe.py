import numpy as np
import cv2 as cv
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import ImageEmbedder
import mediapipe as mp
import os

# Cargar el modelo .tflite solo una vez
MODEL_PATH = "/home/jorge/VIA/umucv/code/DL/mp_embedder/embedder.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Modelo TFLite no encontrado: {MODEL_PATH}")


base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True, quantize=False)
embedder = ImageEmbedder.create_from_options(options)

def prepare(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = embedder.embed(mp_img)
    return np.array(result.embeddings[0].embedding)


def compare(e1, e2):
    return np.linalg.norm(e1 - e2)  # distancia euclídea
