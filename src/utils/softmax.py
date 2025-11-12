import math

def softmax(x):
    if isinstance(x[0], list):  # if x is a matrix (list of lists)
        return [softmax(row) for row in x]
    else:  # x is a vector
        max_x = max(x) # for numerical stability
        exps = [math.exp(i - max_x) for i in x]
        sum_of_exps = sum(exps)
        return [j / sum_of_exps for j in exps]
