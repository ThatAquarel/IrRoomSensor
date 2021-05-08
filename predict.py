import matplotlib.pyplot as plt
import numpy as np
from keract import get_activations
from mpl_toolkits.axes_grid1 import ImageGrid

from helpers import preprocess, postprocess
# from helpers import preprocess, postprocess, split_data
from model import model_arch
from train import checkpoint_path, images, bounding_boxes
from debug_activations import ActivationsPlot


def main():
    x, y = preprocess(images, bounding_boxes)

    # x, y = train_x, train_y

    # train_x, train_y, test_x, test_y = split_data(x_, y_)

    model = model_arch()
    model.compile("adam", "mse", metrics=['accuracy', 'mse'])

    model.load_weights(checkpoint_path)

    x_ = [x[0]]
    x_ = np.array(x_)

    activations = get_activations(model, x_, auto_compile=False)
    ap = ActivationsPlot()
    ap.display(activations, 0)

    pred_y = model.predict(x)

    pred_y = postprocess(pred_y)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(test_x[0])
    # ax[1].imshow(pred_y[0].T)
    # ax[2].imshow(np.reshape(test_y[0], (8, 8)).T)
    #
    # plt.show()

    display = 4

    for i in range(x.shape[0] // display):
        j = i * display

        fig, ax = plt.subplots(display, 4)
        fig.suptitle("Input                Prediction                Adjusted                Actual")

        for k in range(display):
            adjusted = np.copy(pred_y[j + k].T)

            adjusted -= np.mean(adjusted)

            upper = np.amax(adjusted)
            mean = np.mean(adjusted)

            lower = upper - (upper - mean) * 0.25

            adjusted = np.where((lower < adjusted) & (adjusted <= upper), 1, np.zeros(shape=adjusted.shape))

            ax[k][0].imshow(x[j + k])
            ax[k][1].imshow(pred_y[j + k].T)
            ax[k][2].imshow(adjusted)
            ax[k][3].imshow(np.reshape(y[j + k], (8, 8)).T)

        plt.show()


if __name__ == '__main__':
    main()
