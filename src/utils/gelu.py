import math

def gelu_scalar(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

def gelu(tensor):
    if isinstance(tensor, list):
        return [gelu(sub_tensor) for sub_tensor in tensor]
    else:
        return gelu_scalar(tensor)
