import numpy as np
import matplotlib.pyplot as plt

image_raw = np.load(file=".\\2-IR.npy")
mean = np.mean(image_raw)
image_raw = np.where(image_raw < 0, mean, image_raw)

image_mod = np.copy(image_raw)

a_min = np.amin(image_mod)
a_max = np.amax(image_mod)

image_mod = (image_mod - a_min) / (a_max - a_min)

image_pow = np.power(image_mod, 10)
image_exp = np.power(10, image_mod)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(image_raw)
ax[1].imshow(image_pow)
ax[2].imshow(image_exp)
plt.show()
