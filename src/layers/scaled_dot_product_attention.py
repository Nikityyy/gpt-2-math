import math
from src.utils.matrix_multiply import matmul
from src.utils.transpose_matrix import transpose_matrix
from src.utils.masked_softmax import masked_softmax
from src.utils.softmax import softmax

def scaled_dot_product_attention(queries, keys, values, mask=None):
    # Check if batched
    if len(queries) > 0 and isinstance(queries[0][0], list):
        dk = len(keys[0][0])
    else:
        dk = len(keys[0])
    scores = matmul(queries, transpose_matrix(keys))
    # Scale scores
    if len(scores) > 0 and isinstance(scores[0][0], list):
        # 3D
        scaled_scores = [[[score / math.sqrt(dk) for score in row] for row in batch] for batch in scores]
    else:
        # 2D
        scaled_scores = [[score / math.sqrt(dk) for score in row] for row in scores]
    
    if mask is not None:
        scaled_scores = masked_softmax(scaled_scores, mask)
    else:
        scaled_scores = softmax(scaled_scores)
    
    output = matmul(scaled_scores, values)
    return output
