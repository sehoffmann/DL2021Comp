import torch
from torch import nn
import torch.nn.functional
import copy
from dlcomp.config import activation_from_config


class LinearBnAct(nn.Module):

    def __init__(self, in_c, out_c, activation, bias=True, bn=True, track_running_stats=True):
        super(LinearBnAct, self).__init__()

        self.linear = nn.Linear(in_c, out_c, bias=bias and not bn)
        self.bn = nn.BatchNorm1d(out_c, affine=bias, track_running_stats=track_running_stats)
        self.act = activation_from_config(activation)


    def forward(self, x):
        out = self.linear(x)
        if self.bn:
            out = self.bn(out)
        out = self.act(out)
        return out


class ConvBnAct(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, padding, activation, bn=True, track_running_stats=True):
        super(ConvBnAct, self).__init__()

        self.conv = nn.Conv2d(
            in_c, 
            out_c,
            kernel,
            stride=stride,
            padding=padding,
            bias = not bn
        )
        self.bn = nn.BatchNorm2d(out_c, track_running_stats=track_running_stats) if bn else None
        self.act = activation_from_config(activation)


    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        out = self.act(out)
        return out


class SelfAttention2D(nn.Module):

    def __init__(self, in_c, H, W, num_heads):
        super(SelfAttention2D, self).__init__()

        self.shape = (in_c, H, W)
        self.attention = nn.MultiheadAttention(in_c, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        
        out = torch.flatten(x, start_dim=2)
        out = out.permute(0, 2, 1)
        out, _ = self.attention(out, out, out, need_weights=False)
        out = out.permute(0, 2, 1)

        C, H, W = self.shape
        out = out.reshape(-1, C, H, W)
        return out


class UpsamplingConv(nn.Module):

    def __init__(self, in_c, out_c, kernel, activation, bn=True, track_running_stats=True):
        super(UpsamplingConv, self).__init__()
        self.in_conv = ConvBnAct(
            in_c,
            in_c,
            kernel,
            stride=1,
            padding=(kernel-1) // 2,
            activation=activation,
            bn=bn,
            track_running_stats=track_running_stats
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.projection = nn.Conv2d(
            in_c, 
            out_c,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        out = self.in_conv(x)
        out = self.upsample(out)
        out = self.projection(out)
        return out