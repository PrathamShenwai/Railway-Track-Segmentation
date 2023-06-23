import os
import sys
import time
import random
import warnings

import numpy as np
from tqdm import tqdm_notebook as tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Specify Image Dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed(seed)

# Load the quantized model
model_path = '/home/pratham/IISC/railway/railway_u_net_model/trained-rail-unet-v2/rail-u-net-model-v2.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

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

    # Convert image to compatible depth
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_norm = img_gray / 255.0
    img_input = np.expand_dims(img_norm, axis=2)

    # Set the input tensor
    input_details = interpreter.get_input_details()
    input_data = np.expand_dims(img_input, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds_img = (output_data > 0.7).astype(np.uint8) * 255

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
