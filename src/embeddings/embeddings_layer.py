from src.utils.add_matrices import add_matrices

def embeddings_layer(token_embeddings, positional_encodings):
    if not token_embeddings:
        return []
    
    seq_len = len(token_embeddings[0]) 
    if any(len(batch_item) != seq_len for batch_item in token_embeddings):
        raise ValueError("All sequences in batch must have equal lengths")
    
    sliced_pos = positional_encodings[:seq_len]
    
    combined_embeddings = add_matrices(token_embeddings, sliced_pos)
    
    return combined_embeddings
