def _layer_norm_1d(x, epsilon=1e-5):
    if not x: return []
    mean = sum(x) / len(x)
    variance = sum((xi - mean) ** 2 for xi in x) / len(x)
    return [(xi - mean) / (variance + epsilon) ** 0.5 for xi in x]

def layer_norm(tensor, epsilon=1e-5):
    if tensor and isinstance(tensor[0], list):
        return [layer_norm(sub_tensor, epsilon) for sub_tensor in tensor]
    else:
        return _layer_norm_1d(tensor, epsilon)
