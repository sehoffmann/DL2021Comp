from typing import overload
from .default import DefaultLoop
import numpy as np
import wandb

class Groundtruth(DefaultLoop):

    def train(self):
        predictions = []
        for i, (X, Y) in enumerate(self.test_dl):
            predictions += [X]

        imgs = np.moveaxis(np.concatenate(predictions), 1, -1)
        test_images = [wandb.Image(img) for img in imgs]
        wandb.log({'test-images': test_images, 'epoch': 1}, commit=False)