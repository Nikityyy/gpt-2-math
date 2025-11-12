from src.utils.matrix_multiply import matmul
from src.utils.transpose_matrix import transpose_matrix

def output_projection(x, token_embedding_matrix):
    weight = transpose_matrix(token_embedding_matrix)
    
    logits = matmul(x, weight)
    
    return logits
