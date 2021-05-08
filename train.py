import datetime
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from dataset_reader import parse_dataset_xml, parse_dataset_npy, parse_coords
# from helpers import bounding_box_normalize, preprocess, split_data
from helpers import bounding_box_normalize, preprocess
from model import model_arch
from debug_activations import DebugCallback

dataset_xml = ".\\dataset_xml"
dataset_npy = ".\\dataset_npy"

images, boxes = parse_dataset_xml(dataset_xml)
images = [np.asarray(Image.fromarray(image).resize((32, 32))) for image in images]
boxes = [parse_coords(box) for box in boxes]

max_detect = max([box.shape[0] for box in boxes])
num_images = len(images)
num_epochs = 128

checkpoint_path = "training/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

bounding_boxes = bounding_box_normalize(boxes, max_detect)

y_scale = 32 / 24
for bounding_box in bounding_boxes:
    bounding_box[:, 1] *= y_scale
    bounding_box[:, 3] *= y_scale


def main():
    global images, bounding_boxes, max_detect, num_images

    train_x, train_y = parse_dataset_npy(dataset_npy)
    train_x = np.reshape(train_x, (train_x.shape[0], 32, 32, 1))
    train_x /= 255
    train_y = np.reshape(train_y, (train_y.shape[0], 64))

    test_x, test_y = preprocess(images, bounding_boxes)

    # train_x, train_y, test_x, test_y = split_data(x_, y_)

    model = model_arch()
    model.compile("adam", "mse", metrics=['accuracy', 'mse'])

    if len(os.listdir(checkpoint_dir)) != 0:
        model.load_weights(checkpoint_path)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_x, train_y,
              epochs=num_epochs,
              validation_data=(test_x, test_y),
              verbose=2,
              callbacks=[checkpoint_callback, tensorboard_callback, DebugCallback(test_x)],
              use_multiprocessing=True,
              workers=16)


if __name__ == '__main__':
    main()
