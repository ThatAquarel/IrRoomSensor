import xml.etree.ElementTree as ElementTree
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import numpy as np
from PIL import Image

x_tags = ["xmin", "xmax"]
y_tags = ["ymin", "ymax"]


def parse_dataset_xml(directory):
    files = dir_files(directory)

    images = [np.asarray(Image.open(join(directory, "{}.jpg".format(f)))) for f in tqdm(files)]
    boxes = [ElementTree.parse(join(directory, "{}.xml".format(f))).getroot() for f in tqdm(files)]

    return images, boxes


def parse_dataset_npy(directory):
    files = dir_files(directory)

    images = np.array([np.asarray(Image.open(join(directory, "{}.jpg".format(f)))) for f in tqdm(files)]).astype(
        "float32")
    predictions = np.array([np.asarray(np.load(join(directory, "{}.npy".format(f)))) for f in tqdm(files)]).astype(
        "float32")

    return images, predictions


def parse_coords(boxes):
    return np.array([[[int(coords.find(tag).text) for tag in x_tags], [int(coords.find(tag).text) for tag in y_tags]]
                     for coords in boxes.findall("./object/bndbox")])


def dir_files(directory):
    files_ = filter(None, [f.split(".")[0] for f in listdir(directory) if
                           isfile(join(directory, f))])
    files = []
    [files.append(file) for file in files_ if file not in files]

    return files
