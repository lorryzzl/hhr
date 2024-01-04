# 16 frames for a 8-second short clip + 16 frames for a 32-second long clip
# No Yale videos included; No buffer added

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications import Xception
import cv2

MAX_SEQ_LENGTH = 16  #20
NUM_FEATURES = 2048
IMG_SIZE = 512
EPOCHS = 20

model = keras.models.load_model('../results/feature_extractor_model/model_Xception_imagenet_070623.h5')
# Use cholec80-pretrained feature extractor:
#model = keras.models.load_model('../results/feature_extractor_model/finetuned_fe_model/Xception_080623.h5')
base_layers = model.get_layer('xception')

#build a feature extractor using the base layers of the model
def build_feature_extractor():
    feature_extractor = base_layers
    preprocess_input = keras.applications.xception.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

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
def get_fe_img(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    fe_img = feature_extractor.predict(np.expand_dims(img, axis=0))[0]
    return fe_img

def get_fe_imgs(vid_fname, time_start_sec, time_end_sec):
    cap = cv2.VideoCapture(vid_fname)
    # Fast track:
    j = time_start_sec
    fe_fast_imgs = []
    while j < time_end_sec:      
        cap.set(cv2.CAP_PROP_POS_MSEC, j*1000)
        j += 0.5
        success, image = cap.read()
        if success:
            fe_fast_img = get_fe_img(image)
            fe_fast_imgs.append(fe_fast_img)
        else:
            #if we can't read the frame, then we just repeat the last frame
            fe_fast_imgs.append(fe_fast_imgs[-1])
        if len(fe_fast_imgs) == MAX_SEQ_LENGTH:
            break
    # Slow track:
    slow_start_sec = time_start_sec - 24
    fe_slow_imgs = []
    k = slow_start_sec
    while k < time_end_sec:    
        cap.set(cv2.CAP_PROP_POS_MSEC, k*1000)
        k += 2
        success, image = cap.read()
        if success:
            fe_slow_img = get_fe_img(image)
            fe_slow_imgs.append(fe_slow_img)
        else:
            #if we can't read the frame, then we just repeat the last frame
            fe_slow_imgs.append(fe_slow_imgs[-1])
        if len(fe_slow_imgs) == MAX_SEQ_LENGTH:
                break
    print(k, time_end_sec)
    return fe_fast_imgs, fe_slow_imgs

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
X_dict = {}
y_dict = {}



# %%
for vid in vid_list:
    # if os.path.isfile(f'../data/npy_data/X_data_{vid}.npy'):
    #     print(f"Skipping {vid} as X_data_{vid}.npy already exists")
        # continue
    print(f"Processing {vid}")
    vid_phases = phases[phases['vid_id'] == vid]
    len_vid_phases = len(vid_phases)
    X_fe_data = []
    y_fe_data = []
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
            fe_fast_imgs, fe_slow_imgs = get_fe_imgs(vid_fname, time_start_sec, time_end_sec)
            full_imgs = np.concatenate([fe_fast_imgs, fe_slow_imgs], axis=0).tolist()
            X_fe_data.append(full_imgs)
            y_fe_data.append(phase)
    X_fe_data = np.array(X_fe_data)
    y_fe_data = np.array(y_fe_data)
    print(f'{vid} done, captured {len(X_fe_data)} clips')
    print('Shape of features of this clip is:', X_fe_data.shape)
    X_dict[vid] = X_fe_data
    y_dict[vid] = y_fe_data
print('Feature extraction finished!')

# %%
from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121, Xception
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

# %%
#split vid_list into train and test
train_vids = ['vid_14', 'vid_21', 'rush_02', 'vid_4', 'vid_18','rush_0', 'vid_24', 'vid_7', 'vid_17', 'vid_12', 
              'vid_20', 'UNC_29', 'vid_22', 'vid_11', 'rush_01', 'vid_13', 'UNC_27', 'vid_0', 'vid_3']
val_vids = ['vid_1', 'vid_6', 'vid_15', 'UNC_26', 'UNC_28']
test_vids = ['vid_19', 'vid_5', 'vid_16', 'vid_23', 'UNC_30']

#train_vids = ['rush_01', 'vid_19', 'vid_21', 'vid_1', 'UNC_26', 'UNC_27', 'vid_22', 'vid_5', 'vid_4', 'vid_20', 'vid_13', 'rush_02', #'vid_12', 'UNC_30', #'vid_0', 'vid_23', 'vid_17', 'vid_14', 'vid_15', 'rush_0']

#val_vids = ['vid_3', 'vid_7', 'UNC_28', 'vid_16']

#test_vids = ['UNC_29', 'vid_24', 'vid_6', 'vid_11', 'vid_18']


print('training data are:', train_vids)
print('validation data are:', val_vids)
print('testing data are:', test_vids)

X_train = np.vstack([X_dict[vid] for vid in train_vids])
y_train = np.hstack([y_dict[vid] for vid in train_vids])
X_val = np.vstack([X_dict[vid] for vid in val_vids])
y_val = np.hstack([y_dict[vid] for vid in val_vids])
X_test = np.vstack([X_dict[vid] for vid in test_vids])
y_test = np.hstack([y_dict[vid] for vid in test_vids])
print('Dataset shapes:')
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
print(len(np.unique(y_train)), len(np.unique(y_val)), len(np.unique(y_test)))

# %%
y_label_dict = {
    'exposure': 0,
    'hiatal_dissec': 1,
    'fundus_mob': 2,
    'eso_mob': 3,
    'hiatal_repair': 4,
    'wrap': 5,
    'sac_excision': 6,
    'peg_placement': 7,
    'other': 8,
    'oob': 9
}
NUM_CLASSES = len(y_label_dict)
#encode the labels using the dictionary
y_train = np.array([y_label_dict[y] for y in y_train])
y_val = np.array([y_label_dict[y] for y in y_val])
y_test = np.array([y_label_dict[y] for y in y_test])


if os.path.exists('../results/extracted_fe_slowfast_xception_070623/X_train.npy') == False:
    np.save('../results/extracted_fe_slowfast_xception_070623/X_train.npy', X_train)
if os.path.exists('../results/extracted_fe_slowfast_xception_070623/X_val.npy') == False:
    np.save('../results/extracted_fe_slowfast_xception_070623/X_val.npy', X_val)
if os.path.exists('../results/extracted_fe_slowfast_xception_070623/X_test.npy') == False:
    np.save('../results/extracted_fe_slowfast_xception_070623/X_test.npy', X_test)
if os.path.exists('../results/extracted_fe_slowfast_xception_070623/y_train.npy') == False:
    np.save('../results/extracted_fe_slowfast_xception_070623/y_train.npy', y_train)
if os.path.exists('../results/extracted_fe_slowfast_xception_070623/y_val.npy') == False:
    np.save('../results/extracted_fe_slowfast_xception_070623/y_val.npy', y_val)
if os.path.exists('../results/extracted_fe_slowfast_xception_070623/y_test.npy') == False:
    np.save('../results/extracted_fe_slowfast_xception_070623/y_test.npy', y_test)
# %%
print('# of clips per class in train set:')
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)