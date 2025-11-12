from .linear_layer import init_random_linear
from src.utils.matrix_multiply import matmul
from src.utils.add_matrices import add_matrices
from src.utils.gelu import gelu

def init_feed_forward(d_model, d_ff):
    W1, b1 = init_random_linear(d_model, d_ff)
    W2, b2 = init_random_linear(d_ff, d_model)
    return ((W1, b1), (W2, b2))

def feed_forward(x, weights):
    (W1, b1), (W2, b2) = weights
    
    # First linear transformation
    hidden = matmul(x, W1)
    hidden_with_bias = add_matrices(hidden, b1)
    
    # GELU activation
    activated = gelu(hidden_with_bias)
    
    # Second linear transformation
    output = matmul(activated, W2)
    output_with_bias = add_matrices(output, b2)
    
    return output_with_bias
