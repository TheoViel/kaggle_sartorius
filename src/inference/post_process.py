import numpy as np


def post_process_preds(
    result, thresholds_conf=0.5, thresholds_mask=0.5, remove_overlap=False, num_classes=3
):
    masks, boxes = [], []

    if not isinstance(thresholds_conf, float):
        assert len(thresholds_conf) == num_classes
    if not isinstance(thresholds_mask, float):
        assert len(thresholds_mask) == num_classes

    # Get image type & associated thresholds
    """ More accurate way
    lens = []
    for c, (boxes_c, masks_c) in enumerate(zip(result[0], result[1])):
        if len(boxes_c):
            thresh_conf = thresholds_conf if isinstance(thresholds_conf, float) else thresholds_conf[c]
            thresh_mask = thresholds_mask if isinstance(thresholds_mask, float) else thresholds_mask[c]  # noqa
            scores = boxes_c[:, -1]
            last = np.argmax(scores < thresh_conf) if np.min(scores) < thresh_conf else len(masks_c)
            lens.append(last)
        else:
            lens.append(0)
    """

    lens = [len(boxes_c) for boxes_c, masks_c in zip(result[0], result[1])][:num_classes]

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

        if c == num_classes - 1:
            break

    if not len(masks):
        return [], [], cell

    masks = np.concatenate(masks)
    boxes = np.concatenate(boxes)

    # Remove overlap
    if remove_overlap:  # will make computation 35s longer
        order = np.argsort(boxes[:, -1])
        masks = masks[order]
        boxes = boxes[order]

        for i in range(1, len(masks)):
            others = masks[:i].max(0)
            masks[i] *= ~others

    return masks, boxes, cell
