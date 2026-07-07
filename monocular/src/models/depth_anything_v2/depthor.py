import torch
import torch.nn as nn
from src.utils.pointadd import batch_add_points, batch_get_feature, linear_regression
from src.utils.set_mde import compute_rel_depth, compute_metric_depth, set_depthanything, set_depthanything_metric

class Depthor(nn.Module):
    def __init__(self,):
        super(Depthor, self).__init__()

        self.depth_anything= set_depthanything(encoder='vits')

    def forward(self, input_data):
        x = input_data['image']  # [b, 3, 480, 640]
        sparse_depth = input_data['sparse_depth']  # [b, 1, 480, 640]
        B, _, H, W = x.shape

        mde_feat, rel_depth, inv_depth = compute_rel_depth(self.depth_anything, x)
        if torch.count_nonzero(sparse_depth) <= 1:
            abs_rel_depth = input_data['gt']
            abs_inv_depth = input_data['gt']
            # print(1)
        else:
            abs_rel_depth = linear_regression(sparse_depth, rel_depth)
            abs_inv_depth = linear_regression(sparse_depth, inv_depth)

        return abs_inv_depth, abs_inv_depth