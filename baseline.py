# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications import Xception
import cv2
from random import shuffle
import os
import pandas as pd

# %%
#split vid_list into train and test
train_vids = ['vid_14', 'vid_21', 'rush_02', 'vid_4', 'vid_18','rush_0', 'vid_24', 'vid_7', 'vid_17', 'vid_12', 
              'vid_20', 'UNC_29', 'vid_22', 'vid_11', 'rush_01', 'vid_13', 'UNC_27', 'vid_0', 'vid_3']
val_vids = ['vid_1', 'vid_6', 'vid_15', 'UNC_26', 'UNC_28']
test_vids = ['vid_19', 'vid_5', 'vid_16', 'vid_23', 'UNC_30']

X_dict = {}
y_dict = {}
path = '../results/extracted_fe_slowfast_xception_070623/'
for data in os.listdir(path):
    data = str(data)
    if data.split('_')[0] == 'X':
        vid_name = data.split('_')[1] + '_' + data.split('_')[2].split('.')[0]
        vid = np.load(path + data)
        label = np.load(path + 'y_' + vid_name + '.npy')
        X_dict[vid_name] = vid
        y_dict[vid_name] = label
        
X_train = np.vstack([X_dict[vid] for vid in train_vids])
y_train = np.hstack([y_dict[vid] for vid in train_vids])
X_val = np.vstack([X_dict[vid] for vid in val_vids])
y_val = np.hstack([y_dict[vid] for vid in val_vids])
X_test = np.vstack([X_dict[vid] for vid in test_vids])
y_test = np.hstack([y_dict[vid] for vid in test_vids])
print('Dataset shapes:')
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
print(len(np.unique(y_train)), len(np.unique(y_val)), len(np.unique(y_test)))

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

# %%
# Shuffle the training data first.
# This baseline model only uses one frame (the last frame) feature.

N = X_train.shape[0]
ind_list = [i for i in range(N)]
shuffle(ind_list)
X_train  = X_train[ind_list, :,:]
y_train = y_train[ind_list,]

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
MAX_SEQ_LENGTH = 16
NUM_FEATURES = 2048
IMG_SIZE = 512
dropout = 0.3
EPOCHS = 20
NUM_CLASSES = 10
classes = 10

def get_baseline_model():
    inputs = keras.Input(shape=(32, 2048))
    inputs_1 = inputs[:, 15, :]
    inputs_1 = layers.Dense(64, activation='relu')(inputs_1)
    inputs_1 = layers.Dropout(dropout)(inputs_1)
    outputs = layers.Dense(classes, activation="softmax")(inputs_1)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# %%
import datetime
today_str = datetime.date.today().strftime('%m%d%y')

def run_experiment():
    filepath = f"../source/slowfast_baseline_checkpoint_{today_str}"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=5, min_delta=1e-8, restore_best_weights=True
    )

    model = get_baseline_model()
    print(model.summary())
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
    )
    #save the model
    model.save(f'../results/models/slowfast_baseline_{today_str}.h5')
    
    model.load_weights(filepath)
    _, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
 
    return model

# %%
model = run_experiment()
print('model training complete')

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

preds = model.predict(X_test)
preds_df = pd.DataFrame(preds, columns=y_label_dict.keys())
true_df = pd.DataFrame(columns = y_label_dict.keys())
for i in range(len(y_test)):
    true_df.loc[i] = np.zeros(len(y_label_dict.keys()))
    true_df.iloc[i, y_test[i]] = 1

preds_df.to_csv(f'../results/preds/preds_slowfast_baseline_{today_str}.csv')
true_df.to_csv(f'../results/preds/true_slowfast_baseline_{today_str}.csv')

# %%
preds = model.predict(X_val)
preds_df = pd.DataFrame(preds, columns=y_label_dict.keys())
true_df = pd.DataFrame(columns = y_label_dict.keys())
for i in range(len(y_val)):
    true_df.loc[i] = np.zeros(len(y_label_dict.keys()))
    true_df.iloc[i, y_val[i]] = 1

preds_df.to_csv(f'../results/preds/val_preds_slowfast_baseline_{today_str}.csv')
true_df.to_csv(f'../results/preds/val_true_slowfast_baseline_{today_str}.csv')