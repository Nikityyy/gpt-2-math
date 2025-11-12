def transpose_matrix(matrix):
    # Check if batched (3D)
    if len(matrix) > 0 and isinstance(matrix[0][0], list):
        # Batched: matrix (batch, m, n) -> (batch, n, m)
        return [[list(row) for row in zip(*batch)] for batch in matrix]
    else:
        # 2D
        return [list(row) for row in zip(*matrix)]
