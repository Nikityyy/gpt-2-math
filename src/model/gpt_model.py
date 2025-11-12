import src.embeddings as embeddings
from .gpt_decoder import gpt_decoder, init_gpt_decoder
from .output_projection import output_projection

def init_gpt_model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
    weights = {
        "token_embeddings": embeddings.token_embeddings.init_random_embeddings(vocab_size, d_model),
        "positional_encodings": embeddings.positional_encoding.sinusoidal_positional_encoding(max_seq_len, d_model),
        "decoder": init_gpt_decoder(num_layers, d_model, num_heads, d_ff)
    }
    return weights

def gpt_model_forward(batch_token_ids, weights, mask=None):
    token_embedding_matrix = weights["token_embeddings"]
    token_embeds = embeddings.token_embeddings.token_embeddings_lookup(token_embedding_matrix, batch_token_ids)
    
    pos_encodings = weights["positional_encodings"]
    x = embeddings.embeddings_layer.embeddings_layer(token_embeds, pos_encodings)
    
    decoder_weights = weights["decoder"]
    decoder_output = gpt_decoder(x, decoder_weights, mask)

    logits = output_projection(decoder_output, token_embedding_matrix)
    
    return logits
