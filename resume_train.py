#%% Imports
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import glob
import numpy as np
import datetime

if 'ipykernel' in sys.argv[0]:
    os.chdir('thinsection')

#%% Setting path to dataset and dataset properties.
data_dir = os.path.join('dataset\\train\\')
image_list = list(glob.glob(data_dir+'*/*.jpg'))
image_count = len(image_list)

classes = [item.split('\\')[-2] for item in image_list]
class_names = list(np.unique(classes))

#%% Training parameters.
BUFFER_SIZE = 400
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHAN = 3
epochs = 1500

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHAN)

# %% Defining image generator from dataset.
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=(1./127.5)-1,
    rotation_range=45,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    )

train_data_gen = image_generator.flow_from_directory(
    directory=data_dir,
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes = class_names,
    subset ='training',
    # save_to_dir='dataset\\aug\\train'
    )

val_data_gen = image_generator.flow_from_directory(
    directory=data_dir,
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes = class_names,
    subset ='validation',
    # save_to_dir='dataset\\aug\\val'
    )

print('Resume training...')
# %% 
with tf.device('/cpu:0'):
    checks = "checkpoints\\"

    model = tf.keras.models.load_model(checks+'best.hdf5')

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.8),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )

    
    filepath_best=checks+"best.hdf5"
    ckp_best = tf.keras.callbacks.ModelCheckpoint(filepath_best,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    save_freq='epoch')

    filepath_latest=checks+"epoch-{epoch:04d}.hdf5"
    ckp_last = tf.keras.callbacks.ModelCheckpoint(filepath_latest,
    monitor='val_accuracy',
    verbose=0,
    save_best_only=False,
    mode='auto',
    save_weights_only=False,
    save_freq=25
    )

    callbacks_list = [ckp_best,ckp_last]

#%%
with tf.device('/cpu:0'):
    model.fit_generator(
        generator=train_data_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data = val_data_gen,
        validation_steps = 10,
        initial_epoch = 800,
        verbose = 2
        )

# %%
