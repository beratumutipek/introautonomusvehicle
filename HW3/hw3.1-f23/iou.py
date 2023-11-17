import numpy as np
import json
from utils import get_data, calculate_iou


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i, j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


if __name__ == "__main__":
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'road1.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_bboxes = np.array(pred_bboxes)

    ious = calculate_ious(gt_bboxes, pred_bboxes)

    # TODO: Add a new key:value pair to the dictionary in predictions.json with "scores": iou
    for i, p in enumerate(predictions):
        if p['filename'] == filename:
            p['scores'] = ious[:, i].tolist()

    # Save the updated predictions back to the file (assuming predictions.json is writable)
    with open('./data/predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
