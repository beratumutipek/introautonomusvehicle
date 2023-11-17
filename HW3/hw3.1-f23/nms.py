import json

from utils import calculate_iou

def nms(predictions):
    """
    non max suppression
    args:
    - predictions [list]: list of prediction dictionaries
    returns:
    - filtered [list]: filtered bboxes and scores
    """

    data = []
    for pred in predictions:
        bb = pred['boxes']
        sc = pred['scores']
        data.append([bb, sc])

    data_sorted = sorted(data, key=lambda k: k[1], reverse=True)
    filtered = []

    # TODO: Implement Non-Maximum Suppression
    for i, box_i in enumerate(data_sorted):
        discard = False
        for j, box_j in enumerate(data_sorted):
            if i != j:
                iou = calculate_iou(box_i[0], box_j[0])

                # Adjust the IoU threshold as needed
                if iou > 0.5:
                    discard = True
                    break

        if not discard:
            filtered.append(box_i)

    return filtered

if __name__ == '__main__':
    with open('./data/predictions.json', 'r') as f:
        predictions = json.load(f)

    filtered = nms(predictions)
    print(filtered)
