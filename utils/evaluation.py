from utils.utils import relu
import numpy as np


def intersection_over_union(dt_bbox, gt_bbox):
    """
    Intersection over Union between two bboxes
    :param dt_bbox: list or numpy array of size (4,) [x0, y0, x1, y1]
    :param gt_bbox: list or numpy array of size (4,) [x0, y0, x1, y1]
    :return : intersection over union
    """
    intersection_bbox = np.array([max(dt_bbox[0], gt_bbox[0]),
                                  max(dt_bbox[1], gt_bbox[1]),
                                  min(dt_bbox[2], gt_bbox[2]),
                                  min(dt_bbox[3], gt_bbox[3])])

    intersection_area = relu(intersection_bbox[2] - intersection_bbox[0]) * relu(
        intersection_bbox[3] - intersection_bbox[1])
    area_dt = (dt_bbox[2] - dt_bbox[0]) * (dt_bbox[3] - dt_bbox[1])
    area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    union_area = area_dt + area_gt - intersection_area

    iou = intersection_area / union_area
    return iou


def evaluate_sample(target_pred, target_true, iou_threshold=0.5):
    """
    For a given sample, gets if each detection is a True Positive or a False positive
    :param target_pred : dictionary describing the predicted detection (should have 'boxes', 'labels', 'scores')
    :param target_true : dictionary describing the ground truth (should have 'boxes', 'labels')
    :param iou_threshold: The treshold on IoU, to judge if a detection is correct or not
    :return : List of dictionaries with same size as detections, each dict have keys
              'score'  and 'TP' (= 1 if true positive and 0 otherwise)
    """
    gt_bboxes = target_true['boxes'].numpy()
    gt_labels = target_true['labels'].numpy()

    dt_bboxes = target_pred['boxes'].numpy()
    dt_labels = target_pred['labels'].numpy()
    dt_scores = target_pred['scores'].numpy()

    results = []
    for detection_id in range(len(dt_labels)):
        dt_bbox = dt_bboxes[detection_id, :]
        dt_label = dt_labels[detection_id]
        dt_score = dt_scores[detection_id]

        detection_result_dict = {'score': dt_score}

        max_IoU = 0
        max_gt_id = -1
        for gt_id in range(len(gt_labels)):
            gt_bbox = gt_bboxes[gt_id, :]
            gt_label = gt_labels[gt_id]

            if gt_label != dt_label:
                continue

            if intersection_over_union(dt_bbox, gt_bbox) > max_IoU:
                max_IoU = intersection_over_union(dt_bbox, gt_bbox)
                max_gt_id = gt_id

        if max_gt_id >= 0 and max_IoU >= iou_threshold:
            detection_result_dict['TP'] = 1
            gt_labels = np.delete(gt_labels, max_gt_id, axis=0)
            gt_bboxes = np.delete(gt_bboxes, max_gt_id, axis=0)

        else:
            detection_result_dict['TP'] = 0

        results.append(detection_result_dict)

    return results
