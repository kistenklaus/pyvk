import torch.nn as nn
import torch.fx as fx
from .target import Target
from .reflect import reflect_model

def export(
        model: nn.Module, 
        target: Target, 
        input_shape: list[int],
        verbose : bool = False
        ) -> None:
    reflected_layers = reflect_model(model, input_shape)
    for layer in reflected_layers:
        print(layer)
