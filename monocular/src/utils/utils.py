import numpy as np
import matplotlib


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_ewmae(work_image, ref_image, max_depth, kappa=0.5):
    """GCMSE --- Gradient Conduction Mean Square Error.

    Computation of the GCMSE. An image quality assessment measurement
    for image filtering, focused on edge preservation evaluation.

    gcmse: float
        Value of the GCMSE metric between the 2 provided images. It gets
        smaller as the images are more similar.
    """
    # Normalization of the images to [0,1] values.
    max_val = ref_image.max()
    ref_image_float = ref_image.astype('float32')
    work_image_float = work_image.astype('float32')
    normed_ref_image = ref_image_float / max_val
    normed_work_image = work_image_float / max_val

    # Initialization and calculation of south and east gradients arrays.
    gradient_S = np.zeros_like(normed_ref_image)
    gradient_E = gradient_S.copy()
    gradient_S[:-1, :] = np.diff(normed_ref_image, axis=0)
    gradient_E[:, :-1] = np.diff(normed_ref_image, axis=1)

    # Image conduction is calculated using the Perona-Malik equations.
    cond_S = np.exp(-(gradient_S / kappa) ** 2)
    cond_E = np.exp(-(gradient_E / kappa) ** 2)

    # New conduction components are initialized to 1 in order to treat
    # image corners as homogeneous regions
    cond_N = np.ones_like(normed_ref_image)
    cond_W = cond_N.copy()
    # South and East arrays values are moved one position in order to
    # obtain North and West values, respectively.
    cond_N[1:, :] = cond_S[:-1, :]
    cond_W[:, 1:] = cond_E[:, :-1]

    # Conduction module is the mean of the 4 directional values.
    conduction = (cond_N + cond_S + cond_W + cond_E) / 4
    conduction = np.clip(conduction, 0., 1.)
    G = 1 - conduction

    # Calculation of the GCMAE value
    valid = np.logical_and(ref_image < max_depth, ref_image > 0.001)
    ewmae = (abs(G[valid] * (normed_ref_image[valid] - normed_work_image[valid]))).sum() / G[valid].sum()
    return ewmae


def compute_errors(gt, pred, gt_ew, pred_ew, max_depth):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    mae = np.mean(np.abs((pred[gt > 0] - gt[gt > 0]) / gt[gt > 0]))

    ewmae = compute_ewmae(pred_ew, gt_ew, max_depth, kappa=0.5)

    return dict(a1=a1, a2=a2, a3=a3, rmse=rmse, mae=mae, ewmae=ewmae
                , log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel)