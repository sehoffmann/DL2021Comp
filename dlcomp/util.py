import torch

def update_ema_model(model, ema_model, alpha):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        p2.data = alpha*p2 + (1-alpha) * p1.data