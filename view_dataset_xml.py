import matplotlib.patches as patches
import matplotlib.pyplot as plt

from dataset_reader import parse_dataset_xml, parse_coords

dataset = ".\\dataset_xml"
images, boxes = parse_dataset_xml(dataset)


def main():
    global images, boxes
    for i, image in enumerate(images):

        fig, ax = plt.subplots()
        im = ax.imshow(image)
        plt.colorbar(im)

        coords = parse_coords(boxes[i])

        for coord in coords:
            x = coord[0]
            y = coord[1]
            box = patches.Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], linewidth=2, edgecolor='r',
                                    facecolor='none')
            ax.add_patch(box)

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
