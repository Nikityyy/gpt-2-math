def _matmul_3d(batch1, batch2):
    if len(batch1[0][0]) != len(batch2[0]):
         raise ValueError("Incompatible matrix dimensions for multiplication.")

    batch_size = len(batch1)
    m = len(batch1[0])
    k = len(batch1[0][0])
    n = len(batch2[0][0])
    
    result = []
    for b in range(batch_size):
        if len(batch1[b][0]) != len(batch2[b]):
             raise ValueError(f"Incompatible matrix dimensions in batch item {b}")
        batch_result = []
        for i in range(m):
            result_row = []
            for j in range(n):
                sum_product = 0
                for kk in range(k):
                    sum_product += batch1[b][i][kk] * batch2[b][kk][j]
                result_row.append(sum_product)
            batch_result.append(result_row)
        result.append(batch_result)
    return result

def matmul(matrix1, matrix2):
    is_3d = len(matrix1) > 0 and isinstance(matrix1[0], list) and isinstance(matrix1[0][0], list)

    batch1 = matrix1 if is_3d else [matrix1]
    is_3d_m2 = len(matrix2) > 0 and isinstance(matrix2[0], list) and isinstance(matrix2[0][0], list)
    batch2 = matrix2 if is_3d_m2 else [matrix2]

    if len(batch1) > 1 and len(batch2) == 1:
        batch2 = [batch2[0]] * len(batch1)

    result_3d = _matmul_3d(batch1, batch2)

    return result_3d if is_3d else result_3d[0]
