def _transpose_3d(batch):
    return [[list(row) for row in zip(*matrix)] for matrix in batch]

def transpose_matrix(matrix):
    is_3d = len(matrix) > 0 and isinstance(matrix[0], list) and isinstance(matrix[0][0], list)
    
    batch = matrix if is_3d else [matrix]
    transposed_batch = _transpose_3d(batch)
    
    return transposed_batch if is_3d else transposed_batch[0]
