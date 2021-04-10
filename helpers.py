import numpy as np


def split_data(x, y):
    j = int(0.8 * x.shape[0])
    return x[:j], y[:j], x[j:], y[j:]


def preprocess(images, bounding_boxes):
    images = np.array(images)

    num_images = images.shape[0]

    x = (images - np.mean(images)) / np.std(images)
    x = np.reshape(x, (num_images, 32, 32, 1))

    # bounding_boxes_rescaled = (np.array(bounding_boxes) / 4).astype("int32")
    bounding_boxes_rescaled = np.around(np.array(bounding_boxes) / 4, decimals=0).astype("int32")
    # y = np.zeros((num_images, 8, 8, 2))
    y = np.zeros((num_images, 8, 8))

    for i, frame in enumerate(bounding_boxes_rescaled):
        for coords in frame:
            bx = coords[0]
            by = coords[1]
            bw = coords[2]
            bh = coords[3]

            if bw != 0 and bh != 0:
                y[i][bx][by] = 1
            # y[i][bx][by] = [bw, bh]

    # y = y / 8
    # y = np.reshape(y, (num_images, 128))
    y = np.reshape(y, (num_images, 64))

    return x, y


def postprocess(pred_y):
    # pred_y = np.reshape(pred_y, (len(pred_y), 8, 8, 2)) * 8
    pred_y = np.reshape(pred_y, (len(pred_y), 8, 8)) * 255

    # pred_bounding_boxes = np.zeros((len(pred_y), 8, 8, 4))
    # for i, frame in enumerate(pred_y):
    #     for tx, x in enumerate(frame):
    #         for ty, y in enumerate(x):
    #             tw = y[0]
    #             th = y[1]
    #
    #             pred_bounding_boxes[i][tx][ty] = [tx, ty, tw, th]
    #
    # return np.reshape(pred_bounding_boxes, (len(pred_y), 64, 4)) * 4

    return pred_y


def bounding_box_normalize(boxes, max_detect):
    bounding_boxes = []
    for box in boxes:
        normalized = [[0, 0, 0, 0] for _ in range(max_detect)]

        for i, coord in enumerate(box):
            x = coord[0]
            y = coord[1]

            x1 = x[0]
            y1 = y[0]
            x2 = x[1] - x[0]
            y2 = y[1] - y[0]

            normalized[i] = [x1, y1, x2, y2]
        bounding_boxes.append(normalized)

    return np.array(bounding_boxes).astype("float64")


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
