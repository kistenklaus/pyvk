import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Dict, Any, Optional, OrderedDict
from enum import Enum, auto
import logging

# Set up logging for warnings
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Reflect")

class LayerType(Enum):
    INPUT = "input"
    CONV2D = "conv2d"
    ACTIVATION = "activation-function"
    MAXPOOL = "maxpool"
    UPSAMPLE = "upsample"
    CONCAT = "concat"
    OUTPUT = "output"

class ActivationFunction(Enum):
    RELU = "relu"

class ActivationParams(Enum):
    TYPE="type" # : ActivationFunction

class MaxPoolParams(Enum):
    KERNEL_SIZE = "kernel_size"
    STRIDE = "stride"

class UpsampleFilterMode(Enum):
    NEAREST = "nearest",

class UpsampleParams(Enum):
    TARGET_SIZE = "scale_factor"
    MODE = "mode"

class Conv2DParams(Enum):
    KERNEL_SIZE = "kernel_size" # (int,int)
    STRIDE = "stride"   # (int,int)
    PADDING = "padding" # (int,int)
    WEIGHTS = "weights" # tensor
    BIAS = "bias"       # tensor

class ReflectedLayer:
    def __init__(
        self, 
        name: str, 
        layer_type: LayerType, 
        input_shape: Optional[List[int]] = None, 
        output_shape: Optional[List[int]] = None,
        parameters: Optional[Dict[Enum, Any]] = None
    ):
        self.name = name
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.parameters = parameters or {}

    def set_parameter(self, param: Enum, value: Any) -> None:
        """Sets a parameter, ensuring it is valid for the layer type."""
        self.parameters[param] = value

    def __repr__(self) -> str:
        params = {p.name: v for p, v in self.parameters.items() if p.name not in ['WEIGHTS', 'BIAS']}
        return (f"ReflectedLayer(name={self.name}, type={self.layer_type.value}, "
                f"input={self.input_shape}, output={self.output_shape}, "
                f"params={params})")


def reflect_model(model: nn.Module, input_shape: List[int]) -> List[ReflectedLayer]:
    reflected_layers = []
    current_shape = input_shape  # Start with the input shape

    # Traverse the FX graph and reflect layers
    traced = fx.symbolic_trace(model)
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            layer = ReflectedLayer(name=node.name, layer_type=LayerType.INPUT)
            layer.input_shape = input_shape  # Input shape is known
            layer.output_shape = input_shape  # Input and output are the same
            reflected_layers.append(layer)
            continue
        
        if node.op == "output":
            layer = ReflectedLayer(name=node.name, layer_type=LayerType.OUTPUT)
            layer.input_shape = current_shape
            layer.output_shape = current_shape  # Final output shape
            reflected_layers.append(layer)
            continue

        if hasattr(model, str(node.target)):
            target = getattr(model, str(node.target))
            if isinstance(target, torch.nn.Conv2d):
                conv2d : torch.nn.Conv2d = target
                inputDims = len(current_shape) - 2 # minus channels and batch size
                
                padding_mode = conv2d.padding_mode
                assert padding_mode == "zeros", "pyvk currently only supports zero padding";

                padding = conv2d.padding
                stride = conv2d.stride
                kernel_size = conv2d.kernel_size

                if isinstance(padding, str):
                    assert padding == "same" or padding == "valid", "pyvk does only support 'valid' or 'same' as padding values for conv2d"
                    if padding == "same":
                        padding = tuple(
                            max((current_shape[i + 2] - 1) * stride[i] + kernel_size[i] - current_shape[i + 2], 0) // 2
                            for i in range(inputDims)
                        )
                        pass
                    elif padding == "valid":
                        padding = (0,) * inputDims
                        pass
                    else:
                        assert False, "Unreachable";

                if isinstance(padding, int):
                    padding = (padding,) * inputDims


                weights = conv2d.weight
                bias = conv2d.bias
                output_channels = conv2d.out_channels
                input_channels = conv2d.in_channels
                assert input_channels == current_shape[1], "pyvk. Invalid amount of input channels!"

                layer = ReflectedLayer(name=node.name, layer_type=LayerType.CONV2D)
                layer.input_shape = current_shape

                dilation = conv2d.dilation if hasattr(conv2d, 'dilation') else (1, 1)
                H_in, W_in = current_shape[2], current_shape[3]
                P_H, P_W = padding
                K_H, K_W = kernel_size
                S_H, S_W = stride
                D_H, D_W = dilation

                H_out = (H_in + 2 * P_H - D_H * (K_H - 1) - 1) // S_H + 1
                W_out = (W_in + 2 * P_W - D_W * (K_W - 1) - 1) // S_W + 1
                
                current_shape = [current_shape[0], output_channels, H_out, W_out]

                layer.output_shape = current_shape


                layer.set_parameter(Conv2DParams.PADDING, padding)
                layer.set_parameter(Conv2DParams.STRIDE, stride)
                layer.set_parameter(Conv2DParams.KERNEL_SIZE, kernel_size)
                layer.set_parameter(Conv2DParams.BIAS, bias)
                layer.set_parameter(Conv2DParams.WEIGHTS, weights)
                reflected_layers.append(layer)
                continue
            if isinstance(target, torch.nn.MaxPool2d):
                maxPool : torch.nn.MaxPool2d = target

                assert maxPool.ceil_mode == False, "pyvk does not support ceil_mode=True."
                assert maxPool.return_indices == False, "pyvk does not support return_indices=True."
                assert maxPool.padding == 0, "pyvk currently does not support padding for max pooling."
                assert maxPool.stride == maxPool.kernel_size, "pyvk currently requires that the kernel_size == stride"
                assert maxPool.dilation == 1, "pyvk only supports dialation=1 for max pooling"
                
                layer = ReflectedLayer(name=node.name, layer_type=LayerType.MAXPOOL)
                layer.input_shape = current_shape
                H_in, W_in = current_shape[2], current_shape[3]
                KH, KW = maxPool.kernel_size if isinstance(maxPool.kernel_size, tuple) else (maxPool.kernel_size, maxPool.kernel_size)
                H_out = H_in // KH
                W_out = W_in // KW
                current_shape = [current_shape[0], current_shape[1], H_out, W_out]


                layer.set_parameter(MaxPoolParams.KERNEL_SIZE, maxPool.kernel_size)
                layer.set_parameter(MaxPoolParams.STRIDE, maxPool.stride)
                reflected_layers.append(layer)
                continue
            if isinstance(target, torch.nn.Upsample):
                upsample : torch.nn.Upsample = target
                
                layer = ReflectedLayer(name=node.name, layer_type=LayerType.UPSAMPLE)
                layer.input_shape = current_shape
                if upsample.size is not None:
                    layer.set_parameter(UpsampleParams.TARGET_SIZE, upsample.size)
                    current_shape = [current_shape[0], current_shape[1], upsample.size[0], upsample.size[1]]
                else:
                    H_in, W_in = current_shape[2], current_shape[3]
                    assert upsample.scale_factor is not None
                    scale_factor = upsample.scale_factor
                    if isinstance(scale_factor, float):
                        scale_factor = (scale_factor,) * 2
                    
                    H_out = H_in * scale_factor[0]
                    W_out = W_in * scale_factor[1]

                    current_shape = [current_shape[0], current_shape[1], H_out, W_out]
                    
                layer.output_shape = current_shape
                reflected_layers.append(layer)

                continue

        if node.op == "call_function":
            assert callable(node.target)
            if node.target == torch.nn.functional.relu:
                layer = ReflectedLayer(name=node.name, layer_type=LayerType.ACTIVATION)
                layer.input_shape = current_shape
                layer.output_shape = current_shape
                layer.set_parameter(ActivationParams.TYPE, ActivationFunction.RELU)
                reflected_layers.append(layer)
                continue

        logger.warning(f"Unrecognized layer type: {node.target}")

    
    return reflected_layers
