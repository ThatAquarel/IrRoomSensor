import numpy as np
import matplotlib.pyplot as plt


def distribute(values_):
    rounded = np.round(values_, decimals=2)
    unique = np.sort(np.unique(rounded))
    y_ = []
    for i in unique:
        j = np.where(rounded == i)
        y_.append(j[0].shape[0])

    return y_


x = np.linspace(0, 1, 101)

values = np.linspace(0, 1, 1000)
y = distribute(values)

plt.plot(x, y)
plt.show()

values = -1 * (1 - values) ** 2 + 1
y = distribute(values)

plt.plot(x, y)
plt.show()
