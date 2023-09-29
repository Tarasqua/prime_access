"""
@tarasqua 26/09/2023
Вспомогательные функции
"""

import numpy as np


def get_iou(ground_truth: np.array, prediction: np.array) -> float:
    """
    prediction :    the coordinate for predict bounding box
    ground_truth :  the coordinate for ground truth bounding box
    return :        the iou score
    """
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], prediction[0])
    iy1 = np.maximum(ground_truth[1], prediction[1])
    ix2 = np.minimum(ground_truth[2], prediction[2])
    iy2 = np.minimum(ground_truth[3], prediction[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = prediction[3] - prediction[1] + 1
    pd_width = prediction[2] - prediction[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou
