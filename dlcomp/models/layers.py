import torch
from torch import nn
import torch.nn.functional
import copy
from dlcomp.config import activation_from_config


class LinearBnAct(nn.Module):

    def __init__(self, in_c, out_c, activation, bias=True, dropout=0, bn=True, track_running_stats=True):
        super(LinearBnAct, self).__init__()

        self.linear = nn.Linear(in_c, out_c, bias=bias and not bn)
        self.bn = nn.BatchNorm1d(out_c, affine=bias, track_running_stats=track_running_stats)
        self.act = copy.deepcopy(activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None


    def forward(self, x):
        out = self.linear(x)
        
        if self.bn:
            out = self.bn(out)
        
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)
        
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
        self.act = copy.deepcopy(activation)


    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        out = self.act(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, padding, activation, bn=True, track_running_stats=True):
        super(ResidualBlock, self).__init__()

        self.stride = stride

        self.conv1 = ConvBnAct(
            in_c, 
            out_c,
            kernel,
            stride=stride,
            padding=padding,
            activation=activation,
            bn=bn,
            track_running_stats=track_running_stats
        )

        self.conv2 = nn.Conv2d(
            out_c, 
            out_c,
            kernel,
            stride=1,
            padding=(kernel-1) // 2,
            bias = not bn
        )

        self.projection = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else None
        self.bn = nn.BatchNorm2d(out_c, track_running_stats=track_running_stats) if bn else None
        self.act = copy.deepcopy(activation)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.projection:
            out = self.projection(x[:,:,::self.stride, ::self.stride]) + out
        else:
            out = x[:,:,::self.stride, ::self.stride] + out

        if self.bn:
            out = self.bn(out)
        
        out = self.act(out)
        return out


class SelfAttention2D(nn.Module):

    def __init__(self, in_c, H, W, num_heads, layer_norm=True):
        super(SelfAttention2D, self).__init__()

        self.shape = (in_c, H, W)
        self.attention = nn.MultiheadAttention(in_c, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_c) if layer_norm else None

    def forward(self, x):
        
        out = torch.flatten(x, start_dim=2)
        out = out.permute(0, 2, 1)
        
        out, _ = self.attention(out, out, out, need_weights=False)
        if self.norm:
            out = self.norm(out)

        out = out.permute(0, 2, 1)

        C, H, W = self.shape
        out = out.reshape(-1, C, H, W)
        return out


class UpsamplingConv(nn.Module):

    def __init__(self, in_c, out_c, kernel, activation, residual=False, bn=True, track_running_stats=True):
        super(UpsamplingConv, self).__init__()

        if residual:
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
        else:
            self.in_conv = ResidualBlock(
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



class AffineTransform(nn.Module):

    def __init__(self, interpolation_mode='bilinear', padding_mode='zeros', output_size=None):
        super(AffineTransform, self).__init__()
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.output_size = output_size

    
    def forward(self, x, theta):
        if self.output_size:
            B,C = x.shape[:2]
            out_size = (B, C) + self.output_size
        else:
            out_size = x.shape

        grid = nn.functional.affine_grid(theta, out_size, align_corners=False)
        transformed =  nn.functional.grid_sample(
            x,
            grid,
            mode = self.interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=False
        )
        
        return transformed