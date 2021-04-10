import os
import xml.etree.ElementTree as ElementTree
from os.path import join

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm

from dataset_reader import parse_dataset, parse_coords

sources = ".\\adjust"
output = ".\\generated_dataset"

images, boxes = parse_dataset(sources)

flips = [0, 1, (0, 1)]
brightnesses = [-96, -64, 64, 96]


def object_writer(root, x, y):
    person = ElementTree.SubElement(root, "object")
    name = ElementTree.SubElement(person, "name")
    name.text = "person"
    pose = ElementTree.SubElement(person, "pose")
    pose.text = "Unspecified"
    truncated = ElementTree.SubElement(person, "truncated")
    truncated.text = "0"
    difficult = ElementTree.SubElement(person, "difficult")
    difficult.text = "0"
    box = ElementTree.SubElement(person, "bndbox")
    x_min = ElementTree.SubElement(box, "xmin")
    x_min.text = str(x[0])
    y_min = ElementTree.SubElement(box, "ymin")
    y_min.text = str(y[0])
    x_max = ElementTree.SubElement(box, "xmax")
    x_max.text = str(x[1])
    y_max = ElementTree.SubElement(box, "ymax")
    y_max.text = str(y[1])

    return root


def main():
    global images, boxes, flips, brightnesses, output

    for i, image in enumerate(tqdm(images)):
        fig, ax = plt.subplots(len(flips), len(brightnesses))

        coords = parse_coords(boxes[i])

        for j, flip in enumerate(flips):
            for k, brightness in enumerate(brightnesses):
                out_image = join(output, "image-{0}-{1}-{2}.jpg".format(i, j, k))
                out_xml = join(output, "image-{0}-{1}-{2}.xml".format(i, j, k))

                tree = ElementTree.parse("raw.xml")
                root = tree.getroot()
                root[1].text = out_image
                root[2].text = join(os.path.abspath(os.getcwd()), output, out_image)

                modified = np.clip(np.flip(image, flip).astype(np.float32) + brightness, 0, 255).astype(np.uint8)

                ax[j][k].imshow(modified)

                for coord in coords:
                    x = np.copy(coord[0])
                    y = np.copy(coord[1])

                    if flip == 0:
                        y = np.sort(24 - y)
                    elif flip == 1:
                        x = np.sort(32 - x)
                    elif flip == (0, 1):
                        x = np.sort(32 - x)
                        y = np.sort(24 - y)

                    box = patches.Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], linewidth=2, edgecolor='r',
                                            facecolor='none')
                    ax[j][k].add_patch(box)

                    root = object_writer(root, x, y)

                xml_file = open(out_xml, "w+")
                xml_file.write(BeautifulSoup(ElementTree.tostring(root, "utf-8"), "html.parser").prettify())
                xml_file.close()

                image_file = Image.fromarray(modified)
                image_file.save(out_image)

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
