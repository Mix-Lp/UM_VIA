#!/usr/bin/env python

#Script borrowed from umucv/code/DL/UNET
import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText, parser, parse
import time
import torch

parser.add_argument('--model', help="name of model to use", type=str, default='cartas.torch')
args = parse()

sdev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(sdev)
device = torch.device(sdev)

from myUNET import *
model = torch.load(args.model, map_location=device, weights_only=False)

for key, orig in autoStream():
    frame = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    cv.imshow('input', frame)

    # Asegurar que las dimensiones son múltiplos de 32
    H, W = frame.shape
    H2, W2 = H - (H % 32), W - (W % 32)
    frame_resized = cv.resize(frame, (W2, H2))

    inputframes = np.array([frame_resized]).reshape(1, 1, H2, W2).astype(np.float32)

    t0 = time.time()
    inputframes = torch.from_numpy(inputframes).to(device)

    [r] = model(inputframes)
    r = np.clip(r[0].detach().cpu().numpy(), 0, 255).astype(np.uint8)
    t1 = time.time()

    r = np.expand_dims(r, 2) / 255
    mix = (cv.resize(orig, (W2, H2)) * r).astype(np.uint8)
    putText(mix, F"{(t1 - t0) * 1000:5.0f} ms")

    cv.imshow('UNET', mix)
