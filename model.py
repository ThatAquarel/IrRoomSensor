import tensorflow as tf
import tensorflow.keras as keras


def model_arch():
    return keras.models.Sequential([
        keras.layers.Conv2D(16, (1, 1), activation="relu", input_shape=(32, 32, 1)),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.SeparableConv2D(32, (1, 1)),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.SeparableConv2D(64, (3, 3)),

        keras.layers.Conv2D(32, (1, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),

        keras.layers.Activation(tf.nn.sigmoid),
        keras.layers.Dense(128),
        keras.layers.Dense(64),
    ])

# acc: 0.55
# val acc: 0.12
# return keras.models.Sequential([
#         keras.layers.Conv2D(16, (1, 1), activation="relu", input_shape=(32, 32, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#
#         keras.layers.Conv2D(32, (1, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#
#         keras.layers.Conv2D(32, (3, 3)),
#
#         keras.layers.Conv2D(16, (1, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Flatten(),
#
#         keras.layers.Activation(tf.nn.sigmoid),
#         keras.layers.Dense(256),
#         keras.layers.Dense(128)])

# acc: 0.57
# val acc: 0.13
# over fitting
# return keras.models.Sequential([
#         keras.layers.Conv2D(16, (1, 1), activation="relu", input_shape=(32, 32, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#
#         keras.layers.Conv2D(32, (1, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#
#         keras.layers.Conv2D(64, (3, 3)),
#
#         keras.layers.Conv2D(32, (1, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Flatten(),
#
#         keras.layers.Activation(tf.nn.sigmoid),
#         keras.layers.Dense(512),
#         keras.layers.Dense(128)])
