import numpy as np
import json
import os
from PIL import Image
from sentence_transformers import SentenceTransformer, util

def verify_image_embedding(image_path: str):
    search = MultimodalSearch()
    img_embed = search.embed_image(image_path)
    print(f"Embedding shape: {img_embed.shape[0]} dimensions")

def image_search(image_path: str):
    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]
    search = MultimodalSearch(movie_list)
    movie_results = search.search_with_image(image_path)

    for index, (doc_id, score) in enumerate(movie_results):
        print(f"{index+1}. {search.docs[doc_id]['title']} (similarity: {score})")
        print(f"{search.docs[doc_id]['description'][:13]}...")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class MultimodalSearch():

    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.docs = documents
        self.texts = [f"{movie['title']}: {movie['description']}" for movie in self.docs]
        self.text_embeds_path = "./cache/text_embeddings.npy"
        if os.path.exists(self.text_embeds_path):
            self.text_embeddings = np.load(self.text_embeds_path)
        else:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True, normalize_embeddings=True)
            np.save(self.text_embeds_path, self.text_embeddings)

    def embed_image(self, image_path):
        img = Image.open(image_path)
        img_embedding = self.model.encode([img], normalize_embeddings=True)
        return img_embedding[0]
    
    def search_with_image(self, image_path:str):
        results = list()
        img_embedding = self.embed_image(image_path)
        for index, embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(img_embedding, embedding)
            results.append((index, score))

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results[:5]
    

        