import matplotlib.pyplot as plt

from dataset_reader import parse_dataset_npy

dataset = ".\\dataset_npy"
images, predictions = parse_dataset_npy(dataset)


def main():
    global images, predictions
    for i, image in enumerate(images):
        fig, ax = plt.subplots(1, 2)

        im = ax[0].imshow(image)
        plt.colorbar(im)

        ax[1].imshow(predictions[i].T)

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
