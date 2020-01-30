#%% Imports
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import glob
import numpy as np
import datetime
from utils import make_gen
from utils import define_model

#%% Setting path to dataset and dataset properties.
##########################################################
checks = "checkpoints\\"
data_dir = os.path.join('dataset\\train\\')
image_list = list(glob.glob(data_dir+'*/*.jpg'))
image_count = len(image_list)

classes = [item.split('\\')[-2] for item in image_list]
class_names = list(np.unique(classes))

#%% Training parameters.
########################################
CONTINUE = False
BUFFER_SIZE = 400
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHAN = 3
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
epochs = 1500
shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHAN)


#%%Data generator and model
#########################################
gens = make_gen.single_folder(data_dir=data_dir,IMG_HEIGHT=IMG_HEIGHT,IMG_WIDTH=IMG_WIDTH,BATCH_SIZE=BATCH_SIZE,class_names=class_names)
train_data_gen = gens[0]
val_data_gen = gens[1]

if CONTINUE == False:
    FIRST_EPOCH = 0
    model = define_model.resnet_model(len(class_names),shape)
elif CONTINUE == True:
    FIRST_EPOCH = X #User defined interger
    model = tf.keras.models.load_model(checks+'best.hdf5')

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
    )


#%%Callbacks
#########################################
checks = "checkpoints\\"
filepath_best=checks+"best-{epoch:04d}.hdf5"
ckp_best = tf.keras.callbacks.ModelCheckpoint(filepath_best,
monitor='val_accuracy',
verbose=1,
save_best_only=True,
mode='max',
save_weights_only=False,
save_freq='epoch')

# filepath_latest=checks+"epoch-{epoch:04d}.hdf5"
# ckp_last = tf.keras.callbacks.ModelCheckpoint(filepath_latest,
# monitor='val_accuracy',
# verbose=1,
# save_best_only=False,
# mode='auto',
# save_weights_only=False,
# save_freq='epoch'
# )

# callbacks_list = [ckp_best,ckp_last]
callbacks_list = [ckp_best]

#%%Train or resume training
#########################################
model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data = val_data_gen,
    validation_steps = STEPS_PER_EPOCH,
    initial_epoch = FIRST_EPOCH
    )

# %%
