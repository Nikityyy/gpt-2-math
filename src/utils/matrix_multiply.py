def matmul(matrix1, matrix2):
    # number of columns in first matrix must equal number of rows in second
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
