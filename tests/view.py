import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

image = np.load("C:\\Users\\xia_t\\Desktop\\Main Folder\\Ir Room "
                "Sensor\\backup\\images\\image-20210325-130249-299136-1-050ac6ea-dd89-44dd-8013-b1f0e927bdd3-IR.npy")
bounding_boxes = np.array([[0, 0, 5, 5], [20, 19, 5, 5]])

fig = plt.figure(tight_layout=True, figsize=(12, 8))
gs = gridspec.GridSpec(2, 2)

ax0 = fig.add_subplot(gs[0, :])
ax0.set_title('Values')
plt.imshow(image, extent=[0, 32, 0, 24], origin='upper', interpolation='None', aspect='auto')

ax0.xaxis.tick_top()
ax0.set_xticks(np.arange(0.5, 32))
ax0.set_yticks(np.arange(0.5, 24))
ax0.set_xticklabels(np.arange(32))
ax0.set_yticklabels(np.flip(np.arange(24), 0))
ax0.set_xticks(np.arange(32), minor=True)
ax0.set_yticks(np.arange(24), minor=True)
ax0.grid(which='minor', color='k', linestyle='-', linewidth=1)
rounded = np.round(image, decimals=1)
for x in range(32):
    for y in range(24):
        y_ = y * -1 + 23
        ax0.text(x + 0.5, y_ + 0.5, rounded[y, x], ha='center', va='center')

ax1 = fig.add_subplot(gs[1, 0])
ax1.set_title('Grayscale')
plt.imshow(image, extent=[0, 32, 0, 24], origin='upper', interpolation='None', aspect='auto', cmap='Greys')

ax2 = fig.add_subplot(gs[1, 1])
ax2.set_title('Rgb')
plt.imshow(image, extent=[0, 32, 0, 24], origin='upper', interpolation='None', aspect='auto')

for box in bounding_boxes:
    w = box[2]
    h = box[3]
    x = box[0]
    y = box[1] * -1 + 24 - h
    ax0.add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=4))
    ax1.add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=4))
    ax2.add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=4))

plt.show()
