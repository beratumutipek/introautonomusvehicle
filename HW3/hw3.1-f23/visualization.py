import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from utils import get_data, calculate_iou

def visual(ground_truth, predictions):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    - predictions [list[dict]]: model predictions
    """
    paths = glob.glob('./data/images/*')

    # mapping to access data faster
    gtdic = {}
    for gt in ground_truth:
        gtdic[gt['filename']] = gt

    # color mapping of classes
    colormap = {2: [1, 0, 0], 1: [0, 1, 0], 4: [0, 0, 1]}

    for path in paths:
        filename = os.path.basename(path)
        img = Image.open(path)

        f, ax = plt.subplots()
        ax.imshow(img)

        # Ground Truth
        gt_bboxes = gtdic.get(filename, {}).get('boxes', [])
        gt_classes = gtdic.get(filename, {}).get('classes', [])
        for cl, bb in zip(gt_classes, gt_bboxes):
            x1, y1, x2, y2 = bb
            rec = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none',
                            edgecolor=colormap[cl], lw=2)
            ax.add_patch(rec)

        # TODO: Add Prediction boxes
        pred_bboxes = []  # Replace with the actual predicted bounding boxes
        pred_classes = []  # Replace with the actual predicted classes
        for cl, bb in zip(pred_classes, pred_bboxes):
            x1, y1, x2, y2 = bb
            rec = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none',
                            edgecolor=colormap[cl], lw=2, linestyle='dashed')  # Adjust linestyle as needed
            ax.add_patch(rec)

        plt.show()

if __name__ == "__main__":
    ground_truth, predictions = get_data()
    visual(ground_truth, predictions)
