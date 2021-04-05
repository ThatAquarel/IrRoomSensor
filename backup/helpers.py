import numpy as np


def bounding_box_normalize(boxes, max_detect):
    bounding_boxes = []
    for box in boxes:
        # normalized = [[0, 0, 0, 0] for _ in range(max_detect)]
        # normalized = []
        #
        # for i, coord in enumerate(box):
        #     # if i >= max_detect:
        #     #     break
        #
        #     x = coord[0]
        #     y = coord[1]
        #
        #     x1 = x[0]
        #     y1 = y[0]
        #     x2 = x[1] - x[0]
        #     y2 = y[1] - y[0]
        #
        #     normalized[i] = [x1, y1, x2, y2]
        #     # normalized.append([x1, y1, x2, y2])
        #
        # bounding_boxes.append(normalized)

        bounding_boxes.append([box[0][0][0], box[0][1][0], box[0][0][1], box[0][1][1]])
    return np.array(bounding_boxes)


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_i = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_i = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_i <= 0 or h_i <= 0:  # no overlap
        return 0
    i = w_i * h_i

    u = w1 * h1 + w2 * h2 - i

    return i / u


def distance(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))
