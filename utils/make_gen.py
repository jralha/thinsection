import tensorflow as tf


def single_folder(data_dir,IMG_HEIGHT,IMG_WIDTH,BATCH_SIZE,class_names,save_aug_imgs=None,seed=123):


    #Class names needs to be a list with class names.


    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=(1./255),
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        # width_shift_range = 0.1,
        # height_shift_range = 0.1,
        # zoom_range = 0.1,
        # rotation_range = 10
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
        save_to_dir=aug_train,
        seed=seed
        )

    val_data_gen = image_generator.flow_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes = class_names,
        subset ='validation',
        save_to_dir=aug_val,
        seed=seed
        )

    return train_data_gen, val_data_gen