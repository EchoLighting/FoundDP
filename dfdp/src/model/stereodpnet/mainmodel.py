from __future__ import print_function

import pdb
import math
import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.utils.data as torch_data
from einops import rearrange

from dfdp.src.loss.loss_selector import loss_selector
from dfdp.src.metric.metric_selector import metric_selector
# from dataloader.loader_selector import loader_selector

from dfdp.src.model.model_selector import optimizer_selector, scheduler_selector
from dfdp.src.model.stereodpnet.modules import feature_extraction, CostVolume, PSMNetHGAggregation, disp_regression
from dfdp.src.model.stereodpnet.normal_module import ANM


class STEREODPNET(pl.LightningModule):
    def __init__(self, option):
        super(STEREODPNET, self).__init__()
        self.save_hyperparameters()
        
        self.option = option
        self.mindisp = option.model.mindisp
        self.maxdisp = option.model.maxdisp
        self.level = option.model.level
        
        # Feature Extractor
        self.feature_extraction = feature_extraction(option)
        
        # Cost Volume
        self.cost_volume = CostVolume(option, self.mindisp, self.maxdisp)
        
        # Cost Aggregation
        self.aggregation = PSMNetHGAggregation(option.model.inplanes)
        
        # Normal Module
        self.normal_estimator = ANM(option, self.mindisp, self.maxdisp) if option.model.predict_normal else None
        
        # defocus-disparity regressor
        self.regression_layer = disp_regression(self.mindisp, self.maxdisp, self.level)
        
        # loss and metric
        self.loss_model = loss_selector(option)
        self.metric_model = metric_selector(option)
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, xl, yr):
        ref_fea = self.feature_extraction(xl)
        target_fea = self.feature_extraction(yr)
            
        # Cost Volume
        cost = self.cost_volume(ref_fea, target_fea)
        
        # Aggregation
        cost_i, cost = self.aggregation(cost)
        
        # Regression
        cost_f, cost_p = self.regression_layer(cost_i)
        
        # Results
        depth = rearrange(cost_f, 'n b h w -> b n h w')

        return depth
    
