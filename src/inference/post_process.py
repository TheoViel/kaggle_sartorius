import numpy as np


def post_process_preds(result, thresholds_conf=0.5, thresholds_mask=0.5, remove_overlap=False):
    masks = []
    boxes = []

    # Get image type & associated thresholds
    lens = [len(boxes_c) for boxes_c, masks_c in zip(result[0], result[1])]
    cell = np.argmax(lens)
    thresh_conf = thresholds_conf if isinstance(thresholds_conf, float) else thresholds_conf[cell]
    thresh_mask = thresholds_mask if isinstance(thresholds_mask, float) else thresholds_mask[cell]

    # Get masks & filter by confidence
    for c, (boxes_c, masks_c) in enumerate(zip(result[0], result[1])):
        scores = boxes_c[:, -1]

        if len(scores):
            last = np.argmax(scores < thresh_conf) if np.min(scores) < thresh_conf else len(masks_c)
            if last > 0:
                masks.append(np.array(masks_c[:last]) > (thresh_mask * 255))
                boxes.append(boxes_c[:last])

    if not len(masks):
        return [], [], cell

    masks = np.concatenate(masks)
    boxes = np.concatenate(boxes)

    # Remove overlap
    if remove_overlap:  # will make metric computation 35s longer.
        order = np.argsort(boxes[:, -1])
        masks = masks[order]
        boxes = boxes[order]

        for i in range(1, len(masks)):
            others = masks[:i].max(0)
            masks[i] *= ~others

    return masks, boxes, cell
