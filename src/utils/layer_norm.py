def layer_norm(x, epsilon=1e-5):
    mean = sum(x) / len(x)
    variance = sum((xi - mean) ** 2 for xi in x) / len(x)
    normalized = [(xi - mean) / (variance + epsilon) ** 0.5 for xi in x]
    return normalized
