# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='InceptionResNetV2')
parser.add_argument('--weights', type=str, default='imagenet')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_neurons', type=int, default=512)
parser.add_argument('--n_dropout', type=float, default=0.2)
parser.add_argument('--lr_1', type=float, default=3e-4)
parser.add_argument('--lr_2', type=float, default=3e-6)
parser.add_argument('--image_size', type=int, default=512, required=False)
parser.add_argument('--batch_size', type=int, default=16, required=False)

args = parser.parse_args()

model_name = args.model_name
weights = args.weights
n_layers = args.n_layers
n_neurons = args.n_neurons
n_dropout = args.n_dropout
lr_1 = args.lr_1
lr_2 = args.lr_2
img_size = args.image_size
batch_size = args.batch_size



# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121, Xception



# %%
train_dir = '../data/frames/train'
val_dir = '../data/frames/val'
test_dir = '../data/frames/test'
num_classes = len(os.listdir(train_dir))

# %%
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, label_mode='categorical', seed=0, image_size=(img_size, img_size), batch_size=batch_size, color_mode='rgb', validation_split=0.2, subset='training')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, label_mode='categorical', seed=0, image_size=(img_size, img_size), batch_size=batch_size, color_mode='rgb', validation_split=0.2, subset='validation')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode='categorical', seed=0, image_size=(img_size, img_size), batch_size=1, color_mode='rgb')


# %%
#Apply data augmentation
preprocessing_model = tf.keras.Sequential()
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomRotation(40))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"))
preprocessing_model.add(
    tf.keras.layers.experimental.preprocessing.RandomFlip(mode="vertical"))



# %%
train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))


# %% [markdown]
# #only apply normalization to validation set
# val_ds = val_ds.unbatch().batch(batch_size)
# val_ds = val_ds.map(lambda images, labels:
#                     (normalization_layer(images), labels))

# %%
if model_name == 'InceptionResNetV2':
    preprocess_fx = tf.keras.applications.inception_resnet_v2.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-IRV2_notop.h5"
    if weights == 'imagenet':
        base_model = InceptionResNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = InceptionResNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'ResNet50':
    preprocess_fx = tf.keras.applications.resnet50.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-ResNet50_notop.h5"
    if weights == 'imagenet':
        base_model = ResNet50(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = ResNet50(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'InceptionV3':
    preprocess_fx = tf.keras.applications.inception_v3.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-InceptionV3_notop.h5"
    if weights == 'imagenet':
        base_model = InceptionV3(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = InceptionV3(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'DenseNet121':
    preprocess_fx = tf.keras.applications.densenet.preprocess_input
    model_dir = "../RadImageNet/models/RadImageNet-DenseNet121_notop.h5"
    if weights == 'imagenet':
        base_model = DenseNet121(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
    elif weights == 'radimagenet':
        base_model = DenseNet121(input_shape=(img_size, img_size, 3), include_top=False, weights=model_dir, pooling='avg')
elif model_name == 'Xception':
    preprocess_fx = tf.keras.applications.xception.preprocess_input
    if weights == 'imagenet':
        base_model = tf.keras.applications.Xception(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet', pooling='avg')
elif model_name == 'BiT':
    import tensorflow_hub as hub
    base_model = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1", trainable=False)
    preprocess_fx = tf.keras.applications.resnet50.preprocess_input



# %%
inputs = keras.Input(shape=(img_size, img_size, 3))
y = preprocess_fx(inputs)
y = base_model(y, training=False)
for i in range(n_layers):
    y = keras.layers.Dense(n_neurons, activation='relu')(y)
    y = keras.layers.Dropout(n_dropout)(y)
outputs = keras.layers.Dense(num_classes, activation='softmax')(y)
model = keras.Model(inputs, outputs)

# %%
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-8, restore_best_weights=True)

# %%

model.compile(
    optimizer=keras.optimizers.Adam(lr_1), 
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()],
)

epochs = 500
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=early_stopping, verbose=1)
print('phase 1 complete')
# %%
#unfreeze all layers and train at lower learning rate
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(lr_2), 
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()],
)
model.fit(train_ds, epochs=500, validation_data=val_ds, callbacks=early_stopping, verbose=1)
print('phase 2 complete')

#save the model
import datetime
today = datetime.date.today()
today_str = today.strftime('%m%d%y')
model.save('../results/models/model_{}_{}_{}.h5'.format(model_name, weights, today_str))
           

# %%
#load test set and normalize using same code
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
cols = os.listdir(test_dir)
preds_df = pd.DataFrame(columns=cols)
trues_df = pd.DataFrame(columns=cols)

count = 0
for images, labels in test_ds:
    pred = model.predict(images)
    label = labels.numpy()
    preds_df.loc[count] = pred[0]
    trues_df.loc[count] = label[0]
    count += 1

preds_df['path'] = test_ds.file_paths
trues_df['path'] = test_ds.file_paths

#save the preds and trues to csv
preds_df.to_csv('../results/preds/preds_{}_{}_{}.csv'.format(model_name, weights, today_str), index=False)
trues_df.to_csv('../results/preds/trues_{}_{}_{}.csv'.format(model_name, weights, today_str), index=False)

#add _pred to the column names of preds_df
preds_df.columns = [str(col) + '_pred' for col in preds_df.columns]
#add _true to the column names of trues_df
trues_df.columns = [str(col) + '_true' for col in trues_df.columns]

scores_df = pd.DataFrame(columns=['accuracy', 'AUROC', 'AUPRC', 'sensitivity', 'specificity'])
for phase in cols:
    #get the predicted probability of each class and the true class for the current phase
    preds = preds_df[phase + '_pred']
    trues = trues_df[phase + '_true']
    #calculate the AUROC
    auroc = roc_auc_score(trues, preds)
    #calculate the AUPRC
    auprc = average_precision_score(trues, preds)
    #calculate the accuracy
    acc = accuracy_score(trues, preds.round())
    #calculate the sensitivity
    sens = recall_score(trues, preds.round())
    #calculate the specificity
    spec = recall_score(trues, preds.round(), pos_label=0)
    #add the scores to the scores_df
    scores_df.loc[phase] = [acc, auroc, auprc, sens, spec]
scores_df.to_csv('../results/scores/scores_{}_{}_{}.csv'.format(model_name, weights, today_str))