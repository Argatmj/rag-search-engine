from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    with open("./data/movies.json") as f:
            movies = json.load(f)
            movie_list = movies["movies"]
    embeddings = semantic_search.load_or_create_embeddings(movie_list)
    print(f"Number of docs:   {len(semantic_search.documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query: str, limit: int = 5):
    semantic_search = SemanticSearch()
    with open("./data/movies.json") as f:
            movies = json.load(f)
            movie_list = movies["movies"]
    semantic_search.load_or_create_embeddings(movie_list)
    score_list = semantic_search.search(query, limit)

    for index, (score, movie) in enumerate(score_list):
        title = movie["title"]
        desc = movie["description"]
        print(f"{index + 1}. {title} (score: {score})")
        print(f" {desc[:10]}\n")

class SemanticSearch():

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("Text is empty!")

        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self):
        movie_rep_list = []
        for document in self.documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document
            movie_rep = f"{document['title']} {document['description']}"
            movie_rep_list.append(movie_rep)

        self.embeddings = self.model.encode(movie_rep_list, show_progress_bar=True)

        np.save("./cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        cache_embed_path = "./cache/movie_embeddings.npy"
        self.documents = documents 
        for document in self.documents:
            doc_id = document["id"]
            self.document_map[doc_id] = document

        if os.path.exists(cache_embed_path):
            self.embeddings = np.load(cache_embed_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(self.documents)    
    
    def search(self, query, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        score_list = list()

        query_embed = self.generate_embedding(query)
        for index, embedding in enumerate(self.embeddings):
            doc = self.documents[index]
            similarity_score =  cosine_similarity(embedding, query_embed)
            score_list.append((similarity_score, doc))

        sorted_score = sorted(score_list, key=lambda x: x[0], reverse=True)
        return sorted_score[:limit]
