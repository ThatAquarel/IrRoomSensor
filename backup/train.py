import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential

from dataset_reader import parse_dataset, parse_coords
from helpers import bounding_box_normalize, iou

# from helpers import bounding_box_normalize, iou, distance

os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'

dataset = ".\\generated_dataset"
images, boxes = parse_dataset(dataset)
boxes = [parse_coords(box) for box in boxes]

# max_detect = max([box.shape[0] for box in boxes])
max_detect = 1
num_images = len(images)
num_epochs = 8192

checkpoint_path = "training/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

bounding_boxes = bounding_box_normalize(boxes, max_detect)


def main():
    global images, bounding_boxes, max_detect

    images = np.array(images)

    x_ = (images.reshape(num_images, -1) - np.mean(images)) / np.std(images)
    y_ = bounding_boxes.reshape(num_images, -1) / 32

    j = int(0.8 * num_images)

    train_x = x_[:j]
    train_y = y_[:j]

    test_x = x_[j:]
    test_y = y_[j:]
    test_images = images[j:]
    test_bounding_boxes = bounding_boxes[j:]

    model = Sequential([
        Dense(256, input_dim=x_.shape[-1], activation=tf.nn.relu),
        Activation('relu'),
        Dropout(0.2),
        Dense(y_.shape[-1])
        # Dense(y_.shape[-1])
    ])
    model.compile('adadelta', 'mse')

    # model.load_weights(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
    model.fit(train_x, train_y,
              epochs=num_epochs,
              validation_data=(test_x, test_y),
              verbose=2,
              callbacks=[checkpoint_callback])

    pred_y = model.predict(test_x)
    pred_bounding_boxes = pred_y * 32
    pred_bounding_boxes = pred_bounding_boxes.reshape(len(pred_bounding_boxes), max_detect, -1)

    plt.figure(figsize=(12, 3))
    for i_subplot in range(1, 5):
        plt.subplot(1, 4, i_subplot)
        i = np.random.randint(len(test_x))
        plt.imshow(test_images[i], interpolation='none', origin='lower')
        for pred_bbox, exp_bbox in zip(pred_bounding_boxes[i], test_bounding_boxes[i]):
            plt.gca().add_patch(
                patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2] - pred_bbox[0],
                                  pred_bbox[3] - pred_bbox[1], ec='r', fc='none'))

    plt.show()

    summed_iou = 0.
    for pred_bbox, test_bbox in zip(pred_bounding_boxes.reshape(-1, 4), test_bounding_boxes.reshape(-1, 4)):
        summed_iou += iou(pred_bbox, test_bbox)
    mean_iou = summed_iou / len(pred_bounding_boxes)
    print(mean_iou)


if __name__ == '__main__':
    main()
