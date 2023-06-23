import os
import sys
import time
import random
import warnings

import numpy as np
from tqdm import tqdm_notebook as tqdm

from edgetpu.detection.engine import DetectionEngine
from PIL import Image

import cv2

# Specify Image Dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed(seed)

# Load the quantized model
model_path = '/home/pratham/IISC/railway/railway_u_net_model/trained-rail-unet-v2/rail-u-net-model-v2.tflite'
engine = DetectionEngine(model_path)

# Open window to visualize the segmentation
cv2.namedWindow("preview")
cv2.namedWindow("normal")
vc = cv2.VideoCapture(0)

# Try to get the first frame
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# Get the dimensions of the normal window
window_width = frame.shape[1]
window_height = frame.shape[0]

# Set the preview window size to match the normal window
cv2.namedWindow("preview", cv2.WINDOW_NORMAL)


# Initialize FPS variables
start_time = time.time()
frame_count = 0

# Loop until the user presses the ESC key
while rval:
    rval, frame = vc.read()
    ima1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(ima1, (IMG_WIDTH, IMG_HEIGHT))

    # Perform image segmentation with Edge TPU
    pil_img = Image.fromarray(img)
    input_tensor = np.expand_dims(pil_img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    preds_img = (output_tensor > 0.7).astype(np.uint8) * 255

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("preview", np.squeeze(preds_img, axis=0))
    cv2.imshow("normal", frame)
    key = cv2.waitKey(20)
    if key == 27:  # Exit on ESC
        break

# Close windows
cv2.destroyWindow("preview")
cv2.destroyWindow("normal")
