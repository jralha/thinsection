#%% Imports
import os
import sys
import tensorflow as tf
import glob
import random
import numpy as np
import datetime
from utils import make_gen
from utils import define_model
import argparse

#Parsing args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='-1')
parser.add_argument('--continue_training', type=bool, default=False)
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
parser.add_argument('--logs_dir', type=str, default='./logs')
parser.add_argument('--format', type=str, default='jpg')
parser.add_argument('--img_height', type=int, default=256)
parser.add_argument('--img_width', type=int, default=256)
parser.add_argument('--img_chan', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epoch_count', type=int, default=1500)
parser.add_argument('--init_epoch', type=int, default=0)
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--k_fold', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='adam')

parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()

#%% Setting path to dataset and dataset properties.
##########################################################
checks = args.checkpoints_dir+"\\"
data_dir = os.path.join(args.data_dir)
image_list = list(glob.glob(data_dir+'*/*.jpg'))
image_count = len(image_list)

classes = [item.split('\\')[-2] for item in image_list]
class_names = list(np.unique(classes))

#%% Training parameters.
########################################
RUN_NAME = args.run_name
CONTINUE = args.continue_training
BATCH_SIZE = args.batch_size
IMG_WIDTH = args.img_width
IMG_HEIGHT = args.img_height
IMG_CHAN = args.img_chan
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
epochs = args.epoch_count
shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHAN)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

#%%Data generator and model
#########################################
kfold=args.k_fold
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
    if args.model == 'resnet':
        model = define_model.resnet_model(len(class_names),shape)
    elif args.model == 'vgg':
        model = define_model.vgg_model(len(class_names),shape)
    else:
        model = define_model.cnn_shallow(len(class_names),shape)
    
elif CONTINUE == True:
    modelfile = 'best-0677.hdf5'
    FIRST_EPOCH = int(modelfile.split('.')[0].split('-')[-1])
    model = tf.keras.models.load_model(checks+modelfile)
else:
    print('Either start or continue training')
    sys.exit()

model.compile(
    optimizer=args.optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
    )


#%%Callbacks
#########################################
for fold in range(kfold):
    val_acc = '{val_accuracy}'
    if kfold > 1:
        filepath_best=checks+RUN_NAME+"-{epoch}-"+val_acc+"-fold-"+str(fold+1)+"-seed-"+str(seeds[fold])+".hdf5"
    elif kfold == 1:
        filepath_best=checks+RUN_NAME+"-{epoch}-"+val_acc+"-seed-"+str(seeds[fold])+".hdf5"

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

    logfile=RUN_NAME+'.csv'
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
  