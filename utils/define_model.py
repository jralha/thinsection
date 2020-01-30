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
