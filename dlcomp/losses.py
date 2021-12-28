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
