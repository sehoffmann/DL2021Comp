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
        self.p = cfg.pop('p') #torch.nn.Parameter(torch.Tensor([cfg.pop('p')]).to(torch.float32), requires_grad=False)
        self.nan = torch.nn.Parameter(torch.Tensor([float('nan')]).to(torch.float32), requires_grad=False)
    
    def forward(self, input, target):
        mse = self.mse(input * 255, target * 255)  # BCHW
        mse_pixel = torch.mean(mse, axis=1).flatten()  # B x H*W
        sorted, _ = torch.sort(mse_pixel)
        quantile_idx = int((1.-self.p) * sorted.shape[0])
        upper_part = sorted[quantile_idx:]
        return torch.mean(upper_part)

