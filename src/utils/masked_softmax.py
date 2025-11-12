from .softmax import softmax

def masked_softmax(scores, mask):
    is_3d_scores = scores and isinstance(scores[0], list) and isinstance(scores[0][0], list)
    is_2d_mask = mask and isinstance(mask[0], list)

    if is_3d_scores and is_2d_mask:
        return [masked_softmax(matrix_2d, mask) for matrix_2d in scores]

    if scores and isinstance(scores[0], list):  # scores is a 2D matrix
        if mask and isinstance(mask[0], list):  # mask is also a 2D matrix
            return [masked_softmax(row, mask_row) for row, mask_row in zip(scores, mask)]
        else:
            return [masked_softmax(row, mask) for row in scores]
    else:
        masked_vector = [v if m else float('-inf') for v, m in zip(scores, mask)]
        return softmax(masked_vector)
