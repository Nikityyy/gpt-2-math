def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension to be added.")
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

def add_matrices(m1, m2):
    if not isinstance(m1, list) or not isinstance(m2, list):
        return m1 + m2
        
    if len(m1) > 0 and isinstance(m1[0], list) and (not isinstance(m2[0], list) or len(m1) != len(m2)):
        return [add_matrices(sub_m1, m2) for sub_m1 in m1]

    if len(m1) != len(m2):
        raise ValueError("Matrices must have the same dimensions to be added.")
    
    return [add_matrices(sub_m1, sub_m2) for sub_m1, sub_m2 in zip(m1, m2)]
