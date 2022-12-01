from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_embeddings(sentences):
    return model.encode(sentences)


if __name__ == "__main__":
    print(get_sentence_embeddings(["this is a test sentence, that should be embedded"]))