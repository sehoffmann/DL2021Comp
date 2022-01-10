import torch
from torch import nn


class KaggleLoss(nn.Module):
    """
    Expect images and labels to be within range [0, 1]
    """

    def __init__(self, reduction='mean'):
        super(KaggleLoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, input, target):
        return self.mse(input * 255, target * 255)


class BootstrapLoss(nn.Module):
    
    def __init__(self, **cfg):
        super(BootstrapLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.p = torch.nn.Parameter(torch.Tensor([cfg.pop('p')]).to(torch.float32), requires_grad=False)
        self.nan = torch.nn.Parameter(torch.Tensor([float('nan')]).to(torch.float32), requires_grad=False)
    
    def forward(self, input, target):
        mse = self.mse(input * 255, target * 255)  # BCHW
        mse_pixel = torch.mean(mse, axis=1).flatten(start_dim=1)  # B x H*W
        median = torch.quantile(mse_pixel, 1-self.p)
        bootstrapped = torch.where(mse_pixel > median, mse_pixel, self.nan)
        return torch.nanmean(bootstrapped)

