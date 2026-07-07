import torch
import torch.nn as nn
from src.utils.pointadd import batch_add_points, batch_get_feature, linear_regression
from src.utils.set_mde import compute_rel_depth, compute_metric_depth, set_depthanything, set_depthanything_metric

class Depthor(nn.Module):
    def __init__(self,):
        super(Depthor, self).__init__()

        self.depth_anything_metric = set_depthanything_metric(encoder='vitl')

    def forward(self, input_data):
        x = input_data['image']  # [b, 3, 480, 640]
        sparse_depth = input_data['sparse_depth']  # [b, 1, 480, 640]
        B, _, H, W = x.shape

        metric_depth = compute_metric_depth(self.depth_anything_metric, x)
        abs_metric_depth = linear_regression(sparse_depth, metric_depth)

        return metric_depth, metric_depth