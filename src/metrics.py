import numpy as np


def mpjpe_3d(y_true, y_pred, num_joints=17):
    y_true = np.asarray(y_true).reshape((-1, num_joints, 3))
    y_pred = np.asarray(y_pred).reshape((-1, num_joints, 3))
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=2)))


def hpe_li_pck_mmfi(y_true, y_pred, threshold, eps=1e-8):
    """HPE-Li MM-Fi PCK: 2D joint error normalized by shoulder-hip scale.

    This follows HPE-Li-ECCV2024's compute_pck_pckh for MM-Fi.
    The similarly named compute_pck_pckh_18 is used for WiPose, not MM-Fi.

    HPE-Li MM-Fi details:
    - use x/y only,
    - normalize by distance between keypoints 1 and 11,
    - report percentage over all joints and samples.
    """
    y_true = np.asarray(y_true).reshape((-1, 17, 3))[:, :, :2]
    y_pred = np.asarray(y_pred).reshape((-1, 17, 3))[:, :, :2]
    scale = np.linalg.norm(y_true[:, 1, :] - y_true[:, 11, :], axis=1)
    scale = np.maximum(scale, eps)
    dist = np.linalg.norm(y_pred - y_true, axis=2) / scale[:, None]
    return float(100.0 * np.mean(dist <= threshold))
