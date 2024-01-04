# 16 frames for a 8-second short clip + 16 frames for a 32-second long clip
# No Yale videos included; No buffer added
# Extract frames, not features.

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications import Xception
import cv2
import gc

MAX_SEQ_LENGTH = 16  #20
NUM_FEATURES = 2048
IMG_SIZE = 512
EPOCHS = 20

#load the labels
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil

#time is in the form XXH:XXm:XXs, so we need to convert it to seconds
phases = pd.read_excel('../data/phases_use.xlsx', engine='openpyxl')

# %%
phases = phases[['vid_id', 'path', 'phase', 'time_start_sec', 'time_end_sec']]


# %%
MAX_SEQ_LENGTH = 16 #20
NUM_FEATURES = 2048
IMG_SIZE = 512
EPOCHS = 20
len_phases = len(phases)
print("Phases_use has rows:", len_phases)

# %%
def get_prep_img(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def get_prep_imgs(vid_fname, time_start_sec, time_end_sec):
    cap = cv2.VideoCapture(vid_fname)
    # Fast track:
    j = time_start_sec
    prep_fast_imgs = []
    while j < time_end_sec:      
        cap.set(cv2.CAP_PROP_POS_MSEC, j*1000)
        j += 0.5
        success, image = cap.read()
        if success:
            prep_fast_img = get_prep_img(image)
            prep_fast_imgs.append(prep_fast_img)
        else:
            #if we can't read the frame, then we just repeat the last frame
            prep_fast_imgs.append(prep_fast_imgs[-1])
        if len(prep_fast_imgs) == MAX_SEQ_LENGTH:
            break
    # Slow track:
    slow_start_sec = time_start_sec - 24
    prep_slow_imgs = []
    k = slow_start_sec
    while k < time_end_sec:    
        cap.set(cv2.CAP_PROP_POS_MSEC, k*1000)
        k += 2
        success, image = cap.read()
        if success:
            prep_slow_img = get_prep_img(image)
            prep_slow_imgs.append(prep_slow_img)
        else:
            #if we can't read the frame, then we just repeat the last frame
            prep_slow_imgs.append(prep_slow_imgs[-1])
        if len(prep_slow_imgs) == MAX_SEQ_LENGTH:
                break
    print(k, time_end_sec)
    return prep_fast_imgs, prep_slow_imgs

# %%
#define a function that will break up a video into clips of length MAX_SEQ_LENGTH, and return a list of time_start_sec and time_end_sec for each clip that can be used by get_fe_imgs
#with MAX_SEQ_LENGTH of 16, and a frame every 0.5 seconds, this will give us 8 seconds of video for each clip
def get_clips(phase, time_start_sec, time_end_sec):
    MAX_SEQ_LENGTH = 16
    # no buffer added this time
    total_time = time_end_sec - time_start_sec
    num_clips = int(total_time / (MAX_SEQ_LENGTH*0.5))
    clips = []
    for i in range(num_clips):
        clip_start = time_start_sec + i*MAX_SEQ_LENGTH*0.5
        clip_end = clip_start + MAX_SEQ_LENGTH*0.5
        clips.append((clip_start, clip_end))
    return clips

# %%
vid_list = phases['vid_id'].unique()

# %%
for vid in vid_list:
    # if os.path.isfile(f'../data/npy_data/X_data_{vid}.npy'):
    #     print(f"Skipping {vid} as X_data_{vid}.npy already exists")
        # continue
    print(f"Processing {vid}")
    vid_phases = phases[phases['vid_id'] == vid]
    len_vid_phases = len(vid_phases)
    X_img_data = []
    y_img_data = []
    for i in vid_phases.index:
        vid_id = vid_phases.loc[i, 'vid_id']
        vid_fname = vid_phases.loc[i, 'path']
        phase = vid_phases.loc[i, 'phase']
        time_start_sec = vid_phases.loc[i, 'time_start_sec']
        time_end_sec = vid_phases.loc[i, 'time_end_sec']
        clip_list = get_clips(phase, time_start_sec, time_end_sec)
        for clip in clip_list:
            time_start_sec = clip[0]
            time_end_sec = clip[1]
            prep_fast_imgs, prep_slow_imgs = get_prep_imgs(vid_fname, time_start_sec, time_end_sec)
            full_imgs = np.concatenate([prep_fast_imgs, prep_slow_imgs], axis=0).tolist()
            X_img_data.append(full_imgs)
            y_img_data.append(phase)
    X_img_data = np.array(X_img_data)
    y_img_data = np.array(y_img_data)
    if os.path.exists(f'../data/extracted_img_slowfast/X_{vid}.npy') == False:
        np.save(f'../data/extracted_img_slowfast/X_{vid}.npy', X_img_data)
    if os.path.exists(f'../data/extracted_img_slowfast/y_{vid}.npy') == False:
        np.save(f'../data/extracted_img_slowfast/y_{vid}.npy', y_img_data)
    print(f'{vid} done, captured {len(X_img_data)} clips')
    print('Shape of this clip is:', X_img_data.shape)
    del X_img_data
    del y_img_data
    gc.collect()
print('Image extraction finished!')

