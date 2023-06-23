# Import necessary libraries
import os
import sys
import time
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Specify Image Dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed(seed)

# Define mean IOU metric, if needed
def mean_iou(y_true, y_pred):
   prec = []
   for t in np.arange(0.5, 1.0, 0.05):
       y_pred_ = tf.cast(y_pred > t, tf.int32)
       score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
       K.get_session().run(tf.local_variables_initializer())
       with tf.control_dependencies([up_opt]):
           score = tf.identity(score)
       prec.append(score)
   return K.mean(K.stack(prec), axis=0)

# Load the trained model
model = load_model('/home/pratham/IISC/railway/railway_u_net_model/trained-rail-unet-v2/rail-u-net-model-v2.h5', custom_objects={'mean_iou': mean_iou})

# Open video file
video_path = 'railvideo.mp4'
vc = cv2.VideoCapture(video_path)

# Try to get the first frame
if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False

# Get the dimensions of the video frames
window_width = frame.shape[1]
window_height = frame.shape[0]

# Set the preview window size to match the video frame size
cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("preview", window_width, window_height)

# Loop until the video ends
while rval:
    rval, frame = vc.read()
    if rval:
        ima1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize(ima1, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        
        # Convert image to compatible depth
        img = img.astype(np.uint8)
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_norm = img_gray / 255.0
        img_input = np.expand_dims(img_norm, axis=2)
        newframe = np.expand_dims(img_input, axis=0)
        preds = model.predict(newframe)
        preds_img = (preds > 0.7).astype(np.uint8)
        preds_img *= 255  # Scale the prediction to 0-255 range
        cv2.imshow("preview", np.squeeze(preds_img, axis=0))
        cv2.imshow("normal", frame)
        key = cv2.waitKey(20)
        if key == 27:  # Exit on ESC
            break
    else:
        break
    
# Release video capture
vc.release()

# Close windows
cv2.destroyWindow("preview")
cv2.destroyWindow("normal")
