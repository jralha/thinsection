import tensorflow as tf

def vgg_model(num_classes,IMG_HEIGHT,IMG_WIDTH,IMG_CHAN):

    shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHAN)
    vgg = tf.keras.applications.vgg19.VGG19(
    input_shape=shape,
    include_top=False,
    weights='imagenet')


    vgg.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes,activation='softmax')

    model = tf.keras.Sequential([
        vgg,
        global_average_layer,
        prediction_layer
    ])

    return model