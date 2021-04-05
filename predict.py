import matplotlib.patches as patches
import matplotlib.pyplot as plt

from helpers import preprocess, postprocess, split_data
from model import model_arch
from train import checkpoint_path, images, bounding_boxes


def main():
    x_, y_ = preprocess(images, bounding_boxes)

    train_x, train_y, test_x, test_y = split_data(x_, y_)

    model = model_arch()
    model.compile("adam", "mse", metrics=['accuracy', 'mse'])

    model.load_weights(checkpoint_path)

    pred_y = model.predict(test_x)
    pred_bounding_boxes = postprocess(pred_y)

    fig, ax = plt.subplots()
    im = ax.imshow(test_x[0])
    plt.colorbar(im)

    for coords in pred_bounding_boxes[0]:
        box = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=2, edgecolor='r',
                                facecolor='none')
        ax.add_patch(box)

    plt.show()


if __name__ == '__main__':
    main()
