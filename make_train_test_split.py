#%%
import glob
import os
import random
import numpy as np
from shutil import copyfile

from sklearn.model_selection import train_test_split

cwdname = 'thinsection'
if 'ipykernel' in sys.argv[0] and cwdname not in os.getcwd():
    os.chdir(cwdname)

#%%
data_dir = os.path.abspath('dataset\\')
image_list = list(glob.glob(data_dir+'\\full\\'+'*/*.jpg'))
image_count = len(image_list)

classes = [item.split('\\')[-2] for item in image_list]

X_train, X_test, y_train, y_test = train_test_split(
    image_list,
    classes,
    test_size=0.2,
    stratify=classes,
    random_state=123
    )


# %%
train_folder = data_dir+'\\train\\'
if not os.path.exists(train_folder):
    os.mkdir(train_folder)

for file in X_train:
    
    label = file.split('\\')[-2]
    img = file.split('\\')[-1]
    if not os.path.exists(train_folder+label):
        os.mkdir(train_folder+label)
    copyfile(file,train_folder+label+'\\'+img)


test_folder = data_dir+'\\test\\'
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

for file in X_test:
    
    label = file.split('\\')[-2]
    img = file.split('\\')[-1]
    if not os.path.exists(test_folder+label):
        os.mkdir(test_folder+label)
    copyfile(file,test_folder+label+'\\'+img)

# %%
