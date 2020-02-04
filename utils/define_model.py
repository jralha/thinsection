import tensorflow as tf

def vgg_model(num_classes, shape):
    vgg = tf.keras.applications.vgg19.VGG19(
    input_shape=shape,
    include_top=False,
    weights='imagenet')


    vgg.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    model = tf.keras.Sequential([ vgg, global_average_layer, prediction_layer ])

    return model


def resnet_model(num_classes, shape):
    resnet = tf.keras.applications.resnet_v2.ResNet50V2(
        input_shape=shape,
        include_top=False,
        weights='imagenet'
    )

    resnet.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    model = tf.keras.Sequential([ resnet, global_average_layer, prediction_layer ])

    return model

def cnn_shallow(num_classes,shape):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout

    model = Sequential()
    model.add(Conv2D(32, (7, 7), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model