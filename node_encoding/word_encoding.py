from gensim.models import KeyedVectors
import numpy as np
from functools import partial

def get_w2v_embedding(w2v_model, word):
    if word in w2v_model.index_to_key():
        return w2v_model.get_vector(word)
    else:
        return np.zeros(100, dtype="float32")

def w2v_embedder(w2v_model):
    return partial(get_w2v_embedding, w2v_model)

if __name__ == "__main__":
    w2v = KeyedVectors.load_word2vec_format('./models/custom_w2v_100d.txt')
    get_embedding = w2v_embedder(w2v)
    print(get_embedding("test"))
    print(get_embedding("msznbvssbs"))