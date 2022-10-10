import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from keract import get_activations
from mpl_toolkits.axes_grid1 import ImageGrid

matplotlib.use('Qt5Agg')


# noinspection PyMethodMayBeStatic
class ActivationsPlot:
    fig = None
    gs = None
    image_artists = []

    def __init__(self):
        pass

    def display(self, activations, epoch):
        if self.fig is None:
            self.fig = plt.figure(figsize=(16, 8))
            self.gs = self.fig.add_gridspec(1, len(activations))
            self.__draw(activations)
            plt.show(block=False)
        else:
            self.__update(activations)
        self.fig.canvas.set_window_title("training epoch {0}".format(epoch))
        self.fig.canvas.mpl_connect('button_press_event', self.__details)
        plt.draw()

    def destroy(self):
        plt.close(self.fig)

    def __draw(self, activations):
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

            grid = ImageGrid(self.fig, self.gs[0, i], nrows_ncols=(rows, cols), axes_pad=0.0, share_all=True)

            artists = []
            for j, ax in enumerate(grid):
                if j == 0:
                    ax.set_title(key, rotation=15)

                if len(values.shape) <= 2:
                    artist = ax.imshow(np.reshape(values[0], (values[0].shape[0], 1)))
                else:
                    artist = ax.imshow(values[0, :, :, j])
                artists.append(artist)
                ax.get_yaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])

            self.image_artists.append(artists)

        self.fig.tight_layout()

    def __update(self, activations):
        for i, key in enumerate(activations):
            values = activations[key]
            artists = self.image_artists[i]

            for j, artist in enumerate(artists):
                if len(values.shape) <= 2:
                    artist.set_data(np.reshape(values[0], (values[0].shape[0], 1)))
                else:
                    artist.set_data(values[0, :, :, j])

    def __details(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        fig, ax = plt.subplots()

        image = np.array(event.inaxes.images[0].get_array())
        if (image.shape[0] // image.shape[1]) > 16:
            image = np.reshape(image, (image.shape[1], image.shape[0]))
        plt.imshow(image)

        rounded = np.round(image, decimals=1)
        for i in range(rounded.shape[0]):
            for j in range(rounded.shape[1]):
                ax.text(j, i, rounded[i, j], ha="center", va="center", color="w")

        fig.tight_layout()
        plt.show(block=False)


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
        self.activations_plot.display(activations, epoch)
        print("<< set breakpoint here")

    def on_train_end(self, logs=None):
        self.activations_plot.destroy()
