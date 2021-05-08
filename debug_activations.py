import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from keract import get_activations
from mpl_toolkits.axes_grid1 import ImageGrid

matplotlib.use('Qt5Agg')


class ActivationsPlot:
    fig = None
    gs = None

    def __init__(self):
        pass

    def display(self, activations):
        if self.fig is None:
            self.fig = plt.figure(figsize=(16, 8))
            self.__draw(activations)
            plt.show(block=False)
            plt.draw()
        else:
            plt.clf()
            self.__draw(activations)
            plt.draw()

    def destroy(self):
        plt.close(self.fig)

    def __draw(self, activations):
        gs = self.fig.add_gridspec(1, len(activations))

        for i, key in enumerate(activations):
            values = activations[key]

            try:
                rows = values.shape[3]
                cols = 1
                if values.shape[3] > 16:
                    rows = values.shape[3] // 2
                    cols = 2
            except IndexError:
                rows = 1
                cols = 1

            grid = ImageGrid(self.fig, gs[0, i], nrows_ncols=(rows, cols), axes_pad=0.0, share_all=True)

            for j, ax in enumerate(grid):
                if j == 0:
                    ax.set_title(key, rotation=15)

                if len(values.shape) <= 2:
                    ax.imshow(np.reshape(values[0], (values[0].shape[0], 1)))
                else:
                    ax.imshow(values[0, :, :, j])
                ax.get_yaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])

        self.fig.tight_layout()

    def __update(self, activations):
        pass


class DebugCallback(keras.callbacks.Callback):
    activations_plot = None
    test_data = None

    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.activations_plot = ActivationsPlot()

    def on_epoch_end(self, epoch, logs=None):
        x = [self.test_data[0]]
        x = np.array(x)

        activations = get_activations(self.model, x, auto_compile=False)
        self.activations_plot.display(activations)
        print("<< set breakpoint here")

    def on_train_end(self, logs=None):
        self.activations_plot.destroy()
