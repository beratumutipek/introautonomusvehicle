import numpy as np

from iou import calculate_ious
from utils import get_data


def precision_recall(ious, gt_classes, pred_classes, iou_threshold=0.5):
    """
    calculate precision and recall
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    - iou_threshold [float]: IoU threshold to determine true positives
    returns:
    - precision [float]
    - recall [float]
    """
    xs, ys = np.where(ious > iou_threshold)

    # calculate true positive and false positive
    tps = 0
    fps = 0
    for x, y in zip(xs, ys):
        if gt_classes[x] == pred_classes[y]:
            tps += 1
        else:
            fps += 1

    # calculate false negatives
    matched_gt = len(np.unique(xs))
    fns = len(gt_classes) - matched_gt

    # calculate precision and recall
    precision = tps / (tps + fps) if (tps + fps) > 0 else 0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0

    return precision, recall


if __name__ == "__main__":
    ground_truth, predictions = get_data()

    # get bboxes array
    filename = 'road1.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]

    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_bboxes = np.array(pred_bboxes)
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]

    ious = calculate_ious(gt_bboxes, pred_bboxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)

    # TODO: Display the precision and recall info
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
