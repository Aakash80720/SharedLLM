from sentence_transformers import SentenceTransformer
class TransformerModel:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)

    def encode(self, sentences: list) -> list:
        return self.model.encode(sentences, convert_to_tensor=True)