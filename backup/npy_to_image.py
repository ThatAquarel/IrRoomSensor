import xml.etree.ElementTree as Et
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

range_max = 255
sources = ".\\images"

files_ = filter(None, [f.split(".")[0] for f in listdir(sources) if
                       isfile(join(sources, f))])
files = []
[files.append(file) for file in files_ if file not in files]

images = [np.load("{0}\\{1}.npy".format(sources, f)) for f in files]
boxes = [Et.parse("{0}\\{1}.xml".format(sources, f)).getroot() for f in files]
figures = ["{0}\\{1}.png".format(sources, f) for f in files]

x_tags = ["xmin", "xmax"]
y_tags = ["ymin", "ymax"]


def main():
    global range_max, images, boxes, figures, x_tags, y_tags

    for i, image in enumerate(images):
        print("{0} {1}".format(i, files[i]))

        image = np.array(image)
        a_min = np.amin(image)
        a_max = np.amax(image)
        image = ((image - a_min) * 255 / (a_max - a_min)).astype("uint8")

        im = Image.fromarray(image)
        im.save(figures[i])


if __name__ == '__main__':
    main()
