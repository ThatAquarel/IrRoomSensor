import xml.etree.ElementTree as ElementTree
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

x_tags = ["xmin", "xmax"]
y_tags = ["ymin", "ymax"]


def parse_dataset(directory):
    files_ = filter(None, [f.split(".")[0] for f in listdir(directory) if
                           isfile(join(directory, f))])
    files = []
    [files.append(file) for file in files_ if file not in files]

    images = [np.asarray(Image.open(join(directory, "{}.jpg".format(f)))) for f in files]
    boxes = [ElementTree.parse(join(directory, "{}.xml".format(f))).getroot() for f in files]

    return images, boxes


def parse_coords(boxes):
    return np.array([[[int(coords.find(tag).text) for tag in x_tags], [int(coords.find(tag).text) for tag in y_tags]]
                     for coords in boxes.findall("./object/bndbox")])
