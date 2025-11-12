import math
from src.utils.matrix_multiply import matmul
from src.utils.transpose_matrix import transpose_matrix
from src.utils.masked_softmax import masked_softmax
from src.utils.softmax import softmax

def _scale_tensor(tensor, scale_factor):
    if isinstance(tensor, list):
        return [_scale_tensor(item, scale_factor) for item in tensor]
    else:
        return tensor / scale_factor

def scaled_dot_product_attention(queries, keys, values, mask=None):
    is_3d = isinstance(queries[0][0], list)
    keys_batch = keys if is_3d else [keys]
    dk = len(keys_batch[0][0])

    scores = matmul(queries, transpose_matrix(keys))
    # Scale scores
    scaled_scores = _scale_tensor(scores, math.sqrt(dk))
    
    if mask is not None:
        attention_weights = masked_softmax(scaled_scores, mask)
    else:
        attention_weights = softmax(scaled_scores)
    
    output = matmul(attention_weights, values)
    return output
