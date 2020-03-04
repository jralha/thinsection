#%%
import os
import tensorflow as tf
import numpy as np
import glob
import random
np.set_printoptions(suppress=True)

#%%
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

IMG_HEIGHT=256
IMG_WIDTH=256
BATCH_SIZE=1

data_dir = 'dataset\\full\\'
image_list = list(glob.glob(data_dir+'*/*.jpg'))
image_count = len(image_list)
classes = [item.split('\\')[-2] for item in image_list]
class_names = list(np.unique(classes))

#%%
mfile = 'checkpoints\\best-0680-bkp-64val-vgg.hdf5'

model = tf.keras.models.load_model(mfile)

seed = random.randint(1,999)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=(1./255)
        )

gen = image_generator.flow_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes = class_names,
        save_to_dir=False,
        seed=seed
        )

real = gen.classes
preds= np.zeros(len(real))
pred_all = model.predict_generator(gen)
for n,p in enumerate(preds):
    preds[n] = list(pred_all[n]).index(np.max(pred_all[n]))

acc = np.sum(((real == preds)*1))/len(preds)

print(acc)
# %%
