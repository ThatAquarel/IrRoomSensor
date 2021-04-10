import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def main():
    interp_map = np.array([
        [1, np.nan, np.nan, 0],
        [np.nan, 1, np.nan, np.nan],
        [np.nan, np.nan, 1, np.nan],
        [0, np.nan, np.nan, 1]
    ])

    indexes = np.where(np.logical_not(np.isnan(interp_map)) == True)
    items = interp_map[indexes[0], indexes[1]]

    f = interpolate.interp2d(indexes[0], indexes[1], items)

    x = np.linspace(0, 7, 8)

    plt.imshow(f(x, x))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
