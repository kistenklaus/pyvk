import torch
import torch.nn as nn
import pyvk

import torch.nn.functional as F

class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        pm = "zeros"
        self.enc0 = nn.Conv2d(3, 3, 3, padding='same', padding_mode=pm)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        a = self.pool(x)
        b = self.upsample(a)
        return torch.cat([a,b])



model = SimpleConvModel()


class DummyTarget(pyvk.Target):
    def generate_code(self):
        pass

target = DummyTarget()
pyvk.export(model, target, input_shape=[1, 3, 224, 224])
