#%%
import glob
import os
import random
import numpy as np
from shutil import copyfile

from sklearn.model_selection import train_test_split

#%%
def split_dataset(datadir,test_size=0.2,ext='jpg',outdir=None,random_state=None):

    data_dir = os.path.abspath(datadir)
    if outdir == None:
        outdir = datadir
    out_dir = os.path.abspath(outdir)
    image_list = list(glob.glob(data_dir+'\\*\\*.'+ext))
    if len(image_list) == 0:
        print('No '+ext+' images found at folder '+data_dir)
        exit()
    image_count = len(image_list)

    classes = [item.split('\\')[-2] for item in image_list]

    X_train, X_test, y_train, y_test = train_test_split(
        image_list,
        classes,
        test_size=test_size,
        stratify=classes,
        random_state=random_state
        )

    train_folder = out_dir+'\\train\\'
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    for file in X_train:
        
        label = file.split('\\')[-2]
        img = file.split('\\')[-1]
        if not os.path.exists(train_folder+label):
            os.mkdir(train_folder+label)
        copyfile(file,train_folder+label+'\\'+img)


    test_folder = out_dir+'\\test\\'
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    for file in X_test:
        
        label = file.split('\\')[-2]
        img = file.split('\\')[-1]
        if not os.path.exists(test_folder+label):
            os.mkdir(test_folder+label)
        copyfile(file,test_folder+label+'\\'+img)

if __name__ == '__main__':

    folder = 'dataset'
    split_dataset(datadir=folder+'\\full',outdir=folder)
# %%
