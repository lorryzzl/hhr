# Fast-track: TCN. Slow-track: Transformer. Image feature reused.

# %%
from tcn import TCN
from tcn import compiled_tcn
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
from keras import Input, Model
from keras.layers import Dense

# %%
#split vid_list into train and test
train_vids = ['vid_14', 'vid_21', 'vid_4', 'vid_18', 'vid_24', 'vid_7', 'vid_17', 'vid_12', 
              'vid_20', 'vid_22', 'vid_11', 'vid_13', 'vid_0', 'vid_3']
val_vids = ['vid_1', 'vid_6', 'vid_15']
test_vids = ['vid_19', 'vid_5', 'vid_16', 'vid_23']

X_dict = {}
y_dict = {}
path = '../results/extracted_fe_slowfast_xception_070623/'
for data in os.listdir(path):
    data = str(data)
    if data.split('_')[0] == 'X':
        vid_name = data.split('_')[1] + '_' + data.split('_')[2].split('.')[0]
        vid = np.load(path + data)
        new_vid = []
        for clip in vid: # add current frame
            cur_frame = clip[15]
            clip = np.append(clip, np.array([cur_frame]), axis=0)
            new_vid.append(clip)
        vid = np.array(new_vid)
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
# This model consists of a self-attention block and a cross-attention (then reduce demensions to 1024 through Dense) block.
# The current frame is reused 

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dense_dim', type=str, default=16)
parser.add_argument('--num_heads', type=str, default=1)
parser.add_argument('--dropout', type=float, default=0.5)

args = parser.parse_args()

dense_dim = args.dense_dim
num_heads = args.num_heads
dropout = args.dropout

# %%
MAX_SEQ_LENGTH = 16
NUM_FEATURES = 512
IMG_SIZE = 512
MY_DENSE_DIM = int(dense_dim)
MY_NUM_HEADS = int(num_heads)
MY_DROPOUT = dropout
EPOCHS = 20
NUM_CLASSES = 10

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


# %%
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = MY_DENSE_DIM
    num_heads = MY_NUM_HEADS
    classes = NUM_CLASSES
    
    inputs = keras.Input(shape=(33, 2048))
    inputs_1 = inputs[:, 0:16, :]
    inputs_2 = inputs[:, 16:32, :]
    inputs_3 = inputs[:, 32, :] # current frame
    
    tcn_fast = TCN(return_sequences=False, nb_filters=256, dropout_rate=MY_DROPOUT)(inputs_1)  # The TCN layers are here.
    tcn_fast = Dense(128)(tcn_fast)
    
    inputs_2 = layers.Dense(512, activation=tf.nn.sigmoid)(inputs_2)
    transformer_slow = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding")(inputs_2)
    transformer_slow = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(transformer_slow)
    transformer_slow = layers.GlobalMaxPooling1D()(transformer_slow)
    transformer_slow = layers.Dense(128, activation='relu')(transformer_slow)
    
    dense_img = layers.Dense(512, activation = 'sigmoid')(inputs_3)
    dense_img = layers.Dropout(MY_DROPOUT)(dense_img)
    dense_img = layers.Dense(128, activation = 'relu')(dense_img)
    
    full_x = layers.Concatenate()([tcn_fast, transformer_slow, dense_img])
    full_x = layers.Dense(64, activation='relu')(full_x)
    full_x = layers.Dropout(MY_DROPOUT)(full_x)
    outputs = layers.Dense(classes, activation="softmax")(full_x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# %%
import datetime
today_str = datetime.date.today().strftime('%m%d%y')

def run_experiment():
    filepath = f"../source/slowfast_tcn_vit_checkpoint_only_vid_{MY_DENSE_DIM}_{MY_NUM_HEADS}_{MY_DROPOUT}_{today_str}"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=15, min_delta=1e-8, restore_best_weights=True
    )

    model = get_compiled_model()
    print(model.summary())
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
    )
    #save the model
    model.save(f'../results/models/slowfast_tcn_vit_only_vid_{MY_DENSE_DIM}_{MY_NUM_HEADS}_{MY_DROPOUT}_{today_str}.h5')
    
    model.load_weights(filepath)
    _, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
 
    return model

# %%
model = run_experiment()
print('model training complete')

# %%
print('dense_dim is:', MY_DENSE_DIM)
print('num_heads is:', MY_NUM_HEADS)
print('dropout is:', MY_DROPOUT)

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

preds_df.to_csv(f'../results/preds/preds_slowfast_tcn_vit_only_vid_{MY_DENSE_DIM}_{MY_NUM_HEADS}_{MY_DROPOUT}_{today_str}.csv')
true_df.to_csv(f'../results/preds/true_slowfast_tcn_vit_only_vid_{MY_DENSE_DIM}_{MY_NUM_HEADS}_{MY_DROPOUT}_{today_str}.csv')

# %%
preds = model.predict(X_val)
preds_df = pd.DataFrame(preds, columns=y_label_dict.keys())
true_df = pd.DataFrame(columns = y_label_dict.keys())
for i in range(len(y_val)):
    true_df.loc[i] = np.zeros(len(y_label_dict.keys()))
    true_df.iloc[i, y_val[i]] = 1

preds_df.to_csv(f'../results/preds/val_preds_slowfast_tcn_vit_only_vid_{MY_DENSE_DIM}_{MY_NUM_HEADS}_{MY_DROPOUT}_{today_str}.csv')
true_df.to_csv(f'../results/preds/val_true_slowfast_tcn_vit_only_vid_{MY_DENSE_DIM}_{MY_NUM_HEADS}_{MY_DROPOUT}_{today_str}.csv')