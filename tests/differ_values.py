import numpy as np
import matplotlib.pyplot as plt

image_raw = np.load(file=".\\2-IR.npy")
mean = np.mean(image_raw)
image_raw = np.where(image_raw < 0, mean, image_raw)

image_mod = np.copy(image_raw)

mean = np.mean(image_mod)
image_mod -= mean
a_max = np.amax(image_mod)
image_mod /= a_max

a_min = np.amin(image_mod)
image_mod -= a_min
a_max = np.amax(image_mod)
image_mod /= a_max

image_mod = np.power(image_mod, 10)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image_raw)
ax[1].imshow(image_mod)
plt.show()
