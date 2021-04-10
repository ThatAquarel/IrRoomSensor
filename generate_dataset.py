from os.path import join

import numpy as np
from PIL import Image
from scipy import interpolate

# import matplotlib.pyplot as plt

divisions = 16
radius_steps = 3
write_dir = ".\\virtual_dataset"


def circle_map(radius, people_x, people_y, people_count, size):
    x = np.reshape(np.repeat(people_x, size), (people_count, size))
    y = np.reshape(np.repeat(people_y, size), (people_count, size))

    angles = np.linspace(0, 2 * np.pi, divisions)
    angles = np.repeat(angles, radius_steps)
    sin = np.sin(angles)
    cos = np.cos(angles)

    opposite = sin * radius + y
    adjacent = cos * radius + x

    return np.reshape(np.dstack((opposite, adjacent)), (people_count * size, 2)).astype(np.int32)


def main():
    image = np.zeros(shape=(32, 32))

    people_count = np.random.randint(0, 5)

    people_coords = np.unique(np.random.randint(0, 4, size=(people_count, 2)), axis=0) * 8.
    people_count = people_coords.shape[0]

    skew = np.random.uniform(-3, 3, size=(people_count, 2))
    people_coords += skew

    people_x = people_coords[:, 0]
    people_y = people_coords[:, 1]

    predict = np.zeros(shape=(8, 8))
    predict_coords = np.round(people_coords / 4, decimals=0).astype(np.int32)
    for coords in predict_coords:
        predict[coords[1], coords[0]] = 1

    if people_count > 0:
        size = divisions * radius_steps

        radius = np.random.randint(0, 4, size=(people_count, divisions))

        radius_adjust = []
        for radius_ in radius:
            radius_adjust.append(np.linspace(0, radius_, radius_steps, axis=1))
        radius_adjust = np.array(radius_adjust)

        radius_small = np.reshape(radius_adjust, (people_count, size))
        radius_large = np.reshape(radius_adjust, (people_count, size)) + 1
        final_coords_small = circle_map(radius_small, people_x, people_y, people_count, size)
        final_coords_large = circle_map(radius_large, people_x, people_y, people_count, size)

        for coords in final_coords_small:
            image[coords[0]][coords[1]] = 1

        for coords in final_coords_large:
            if image[coords[0]][coords[1]] != 1:
                image[coords[0]][coords[1]] = 0.25

    people_mask = image.copy().astype(bool)
    people_skew = np.random.uniform(0, 0.5, size=(32, 32))

    background_mask = np.where(image == 1, -1, image) + 1
    background_skew = np.random.uniform(0, 0.2, size=(32, 32))

    image += people_mask.astype("float32") * people_skew
    image += background_mask.astype("float32") * background_skew

    if people_count > 1:
        interp_x = np.append(people_x, [0, 0, 31, 31])
        interp_y = np.append(people_y, [0, 31, 0, 31])
        interp_z = np.append(np.ones(people_count), np.zeros(4))
        f = interpolate.interp2d(interp_x, interp_y, interp_z)
        interp_map = np.linspace(0, 31, 32)

        interp_skew = f(interp_map, interp_map)
        image += (interp_skew * 0.1)

    a_max = np.amax(image)
    a_min = np.amin(image)

    if a_max > 0.75:
        image /= a_max

    if not a_min < 0 and not a_max > 1.75:
        # plt.imshow(image)
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(predict)
        # plt.colorbar()
        # plt.show()

        return image, predict
    else:
        return None


if __name__ == '__main__':
    # main()

    for i in range(100):
        out = None

        while out is None:
            out = main()

        pil_image = Image.fromarray(np.repeat((out[0].reshape((32, 32, 1)) * 255).astype("uint8"), 3, axis=2))
        pil_image.save(join(write_dir, "image-{0}.jpg".format(i)))

        np.save(join(write_dir, "image-{0}.npy".format(i)), out[1])
