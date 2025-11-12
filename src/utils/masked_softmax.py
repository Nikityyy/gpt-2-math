from . import softmax

def masked_softmax(vector, mask):
    if isinstance(vector[0], list):  # vector is a matrix
        if isinstance(mask[0], list):  # mask is also a matrix
            return [masked_softmax(row, mask_row) for row, mask_row in zip(vector, mask)]
        else:  # mask is a vector applied to each row
            return [masked_softmax(row, mask) for row in vector]
    else:  # vector is a list
        masked_vector = [v if m else float('-inf') for v, m in zip(vector, mask)]
        softmax_vector = softmax.softmax(masked_vector)
        return softmax_vector
