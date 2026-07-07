import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        if concat_with is None:
            up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            f = up_x
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super(Decoder, self).__init__()

        img_channels = [256, 128, 64, 32, 16]
        dep_channels = [256, 128, 64, 32, 16]
        mix_channels = [512, 256, 128, 64, 32]
        decoder_channels = [256, 128, 64, 32]

        self.up4 = UpSampleBN(skip_input=mix_channels[0], output_features=decoder_channels[0])
        self.up3 = UpSampleBN(skip_input=mix_channels[1] + decoder_channels[0], output_features=decoder_channels[1])
        self.up2 = UpSampleBN(skip_input=mix_channels[2] + decoder_channels[1], output_features=decoder_channels[2])
        self.up1 = UpSampleBN(skip_input=mix_channels[3] + decoder_channels[2], output_features=decoder_channels[3])

        self.conv4 = nn.Sequential(
            nn.Conv2d(img_channels[0] + dep_channels[0], mix_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mix_channels[0]),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(img_channels[1] + dep_channels[1], mix_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mix_channels[1]),
            nn.LeakyReLU(),
            # LargeKernel(mix_channels[1], large_kernel=7),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(img_channels[2] + dep_channels[2], mix_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mix_channels[2]),
            nn.LeakyReLU(),
            # LargeKernel(mix_channels[2], large_kernel=11),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels[3] + dep_channels[3], mix_channels[3], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(mix_channels[3]),
            nn.LeakyReLU(),
            # LargeKernel(mix_channels[3], large_kernel=15),
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(img_channels[4] + dep_channels[4], mix_channels[4], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(mix_channels[4]),
            nn.LeakyReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(mix_channels[4] + decoder_channels[3], num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(),
        )

    def forward(self, img_features, hist_features, ):
        x_block0, x_block1, x_block2, x_block3, x_block4 = img_features
        depth_feat0, depth_feat1, depth_feat2, depth_feat3, depth_feat4 = hist_features

        x_d4 = torch.cat([x_block4, depth_feat4], dim=1)  # [b, 128+232, 15, 20]
        x_d4 = self.conv4(x_d4)  # [b, 256, 15, 20]
        x_d4_fused = self.up4(x_d4, None)  # [b, 128, 30, 40]

        x_d3 = torch.cat([x_block3, depth_feat3], dim=1)  # [b, 64+136, 30, 40]
        x_d3 = self.conv3(x_d3)  # [b, 128, 30, 40]
        x_d3_fused = self.up3(torch.cat([x_d4_fused, x_d3],dim=1), None)  # [b, 64, 60, 80]

        x_d2 = torch.cat([x_block2, depth_feat2], dim=1)  # [b, 32+56, 60, 80]
        x_d2 = self.conv2(x_d2)  # [b, 64, 60, 80]
        x_d2_fused = self.up2(torch.cat([x_d3_fused, x_d2], dim=1), None)  # [b, 32, 120, 160]

        x_d1 = torch.cat([x_block1, depth_feat1], dim=1)  # [b, 16+40, 120, 160]
        x_d1 = self.conv1(x_d1)  # [b, 32, 120, 160]
        x_d1_fused = self.up1(torch.cat([x_d2_fused, x_d1], dim=1), None)  # [b, 16, 240, 320]

        x_d0 = torch.cat([x_block0, depth_feat0], dim=1)  # [b, 8+16, 240, 320]
        x_d0 = self.conv0(x_d0)  # [b, 16, 240, 320]
        x_d0_fused = self.conv_out(torch.cat([x_d1_fused, x_d0], dim=1))  # [b, 32, 240, 320]

        return x_d0_fused


class UpSampleBN_S(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN_S, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),)

    def forward(self, x, concat_with):
        if concat_with is None:
            up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            f = up_x
        else:
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Decoder_S(nn.Module):
    def __init__(self, num_classes=1):
        super(Decoder_S, self).__init__()

        img_channels = [256, 128, 64, 32, 16]
        dep_channels = [256, 128, 64, 32, 16]
        mix_channels = [256, 128, 64, 32, 16]
        decoder_channels = [256, 128, 64, 32]

        self.up4 = UpSampleBN_S(skip_input=mix_channels[0], output_features=decoder_channels[0])
        self.up3 = UpSampleBN_S(skip_input=mix_channels[1] + decoder_channels[0], output_features=decoder_channels[1])
        self.up2 = UpSampleBN_S(skip_input=mix_channels[2] + decoder_channels[1], output_features=decoder_channels[2])
        self.up1 = UpSampleBN_S(skip_input=mix_channels[3] + decoder_channels[2], output_features=decoder_channels[3])

        self.conv4 = nn.Sequential(
            nn.Conv2d(img_channels[0] + dep_channels[0], mix_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mix_channels[0]),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(img_channels[1] + dep_channels[1], mix_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mix_channels[1]),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(img_channels[2] + dep_channels[2], mix_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mix_channels[2]),
            nn.LeakyReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels[3] + dep_channels[3], mix_channels[3], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(mix_channels[3]),
            nn.LeakyReLU(),
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(img_channels[4] + dep_channels[4], mix_channels[4], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(mix_channels[4]),
            nn.LeakyReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(mix_channels[4] + decoder_channels[3], num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(),
        )

    def forward(self, img_features, hist_features, ):
        x_block0, x_block1, x_block2, x_block3, x_block4 = img_features
        depth_feat0, depth_feat1, depth_feat2, depth_feat3, depth_feat4 = hist_features

        x_d4 = torch.cat([x_block4, depth_feat4], dim=1)  # [b, 128+232, 15, 20]
        x_d4 = self.conv4(x_d4)  # [b, 256, 15, 20]
        x_d4_fused = self.up4(x_d4, None)  # [b, 128, 30, 40]

        x_d3 = torch.cat([x_block3, depth_feat3], dim=1)  # [b, 64+136, 30, 40]
        x_d3 = self.conv3(x_d3)  # [b, 128, 30, 40]
        x_d3_fused = self.up3(torch.cat([x_d4_fused, x_d3],dim=1), None)  # [b, 64, 60, 80]

        x_d2 = torch.cat([x_block2, depth_feat2], dim=1)  # [b, 32+56, 60, 80]
        x_d2 = self.conv2(x_d2)  # [b, 64, 60, 80]
        x_d2_fused = self.up2(torch.cat([x_d3_fused, x_d2], dim=1), None)  # [b, 32, 120, 160]

        x_d1 = torch.cat([x_block1, depth_feat1], dim=1)  # [b, 16+40, 120, 160]
        x_d1 = self.conv1(x_d1)  # [b, 32, 120, 160]
        x_d1_fused = self.up1(torch.cat([x_d2_fused, x_d1], dim=1), None)  # [b, 16, 240, 320]

        x_d0 = torch.cat([x_block0, depth_feat0], dim=1)  # [b, 8+16, 240, 320]
        x_d0 = self.conv0(x_d0)  # [b, 16, 240, 320]
        x_d0_fused = self.conv_out(torch.cat([x_d1_fused, x_d0], dim=1))  # [b, 32, 240, 320]

        return x_d0_fused


class DepthRegression(nn.Module):
    def __init__(self, in_channels, dim_out=256, embedding_dim=128, norm='linear'):
        super(DepthRegression, self).__init__()
        self.norm = norm
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):                       # [1, 128, 240, 320]
        range_attention_maps = self.conv3x3(x)  # [1, 128, 240, 320]
        regression_head = self.conv1x1(x)       # [1, 128, 240, 320]
        regression_head = regression_head.mean([2,3])  # [1,128]
        y = self.regressor(regression_head)            # [1,256]
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class DepthRegression_S(nn.Module):
    def __init__(self, in_channels, dim_out=256, embedding_dim=64, norm='linear'):
        super(DepthRegression_S, self).__init__()
        self.norm = norm
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):                       # [1, 128, 240, 320]
        range_attention_maps = self.conv3x3(x)  # [1, 128, 240, 320]
        regression_head = self.conv1x1(x)       # [1, 128, 240, 320]
        regression_head = regression_head.mean([2,3])  # [1,128]
        y = self.regressor(regression_head)            # [1,256]
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps