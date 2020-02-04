#%% Imports
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import glob
import random
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
RUN_NAME = 'GPU-RESNET-' #no spaces or dots
CONTINUE = False
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHAN = 3
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
epochs = 1500
shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHAN)


#%%Data generator and model
#########################################
kfold=1
train_data_gen=[]
val_data_gen=[]
seeds=[]
for fold in range(kfold):
    seed = random.randint(1,999)
    seeds.append(seed)
    gens = make_gen.single_folder(data_dir=data_dir,IMG_HEIGHT=IMG_HEIGHT,IMG_WIDTH=IMG_WIDTH,BATCH_SIZE=BATCH_SIZE,class_names=class_names, seed=seed )
    train_data_gen.append(gens[0])
    val_data_gen.append(gens[1])

if CONTINUE == False:
    FIRST_EPOCH = 1
    # model = define_model.cnn_shallow(len(class_names),shape)
    model = define_model.resnet_model(len(class_names),shape)
elif CONTINUE == True:
    modelfile = 'best-0677.hdf5'
    FIRST_EPOCH = int(modelfile.split('.')[0].split('-')[-1])
    model = tf.keras.models.load_model(checks+modelfile)
else:
    print('Either start or continue training')
    sys.exit()

model.compile(
    # optimizer='rmsprop',
    # optimizer=tf.keras.optimizers.Adam(),
    optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=True),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
    )


#%%Callbacks
#########################################
for fold in range(kfold):
    if kfold > 1:
        filepath_best=checks+RUN_NAME+"{epoch}-{val_accuracy}-fold-"+str(fold+1)+"-seed-"+str(seeds[fold])+".hdf5"
    elif kfold == 1:
        filepath_best=checks+RUN_NAME+"{epoch}-{val_accuracy}-seed-"+str(seeds[fold])+".hdf5"

    ckp_best=tf.keras.callbacks.ModelCheckpoint(filepath_best,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False,
        save_freq='epoch'
        )

    log_dir="logs\\"
    # board=tf.keras.callbacks.TensorBoard(log_dir=log_dir,
    #     histogram_freq=1,
    #     write_graph=True
    #     )

    logfile=filepath_best.split('.')[0].split('\\')[-1]
    csv_log=tf.keras.callbacks.CSVLogger(filename=log_dir+logfile)

    callbacks_list = [ckp_best,csv_log]

    #%%Train or resume training
    #########################################

    model.fit_generator(
        generator=train_data_gen[fold],
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=val_data_gen[fold],
        validation_steps=STEPS_PER_EPOCH,
        initial_epoch=FIRST_EPOCH
        )

# %%
  