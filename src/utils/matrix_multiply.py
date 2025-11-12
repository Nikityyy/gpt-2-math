def matmul(matrix1, matrix2):
    # Check if batched (3D)
    if len(matrix1) > 0 and isinstance(matrix1[0][0], list):
        # Batched: matrix1 (batch, m, k), matrix2 (batch, k, n)
        batch_size = len(matrix1)
        m = len(matrix1[0])
        k = len(matrix1[0][0])
        n = len(matrix2[0][0])
        result = []
        for b in range(batch_size):
            batch_result = []
            for i in range(m):
                result_row = []
                for j in range(n):
                    sum_product = 0
                    for kk in range(k):
                        sum_product += matrix1[b][i][kk] * matrix2[b][kk][j]
                    result_row.append(sum_product)
                batch_result.append(result_row)
            result.append(batch_result)
        return result
    else:
        # 2D
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Incompatible matrix dimensions for multiplication.")
        
        result = []
        for i in range(len(matrix1)):
            result_row = []
            for j in range(len(matrix2[0])):
                sum_product = 0
                for k in range(len(matrix2)):
                    sum_product += matrix1[i][k] * matrix2[k][j]
                result_row.append(sum_product)
            result.append(result_row)

        return result
