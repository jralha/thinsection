import tensorflow as tf


def single_folder(data_dir,IMG_HEIGHT,IMG_WIDTH,BATCH_SIZE,class_names,save_aug_imgs=None):


    #Class names needs to be a list with class names.


    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=(1./255),
        rotation_range=45,
        # zoom_range=0.2,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        )

    if save_aug_imgs != None:
        aug_train = 'dataset\\aug\\train'
        aug_val = 'dataset\\aug\\val'
    else:
        aug_train = None
        aug_val = None


    train_data_gen = image_generator.flow_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes = class_names,
        subset ='training',
        save_to_dir=aug_train
        )

    val_data_gen = image_generator.flow_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes = class_names,
        subset ='validation',
        save_to_dir=aug_val
        )

    return train_data_gen, val_data_gen