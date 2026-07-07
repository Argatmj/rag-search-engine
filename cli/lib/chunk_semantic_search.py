from lib.semantic_search import SemanticSearch, cosine_similarity
import numpy as np
import json
import copy 
import re
import os 

def embed_chunks():
    chunk_semantic_search = ChunkSemanticSearch()
    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]

    embeddings = chunk_semantic_search.load_or_create_chunk_embeddings(movie_list)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query: str, limit: int):
    chunk_semantic_search = ChunkSemanticSearch()
    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]
    chunk_semantic_search.load_or_create_chunk_embeddings(movie_list)
    movie_results = chunk_semantic_search.search_chunks(query, limit)

    for idx, movie in enumerate(movie_results):
        print(f"\n{idx}. {movie['title']} (score: {movie['score']})")
        print(f"   {movie['description']}...")
    
class ChunkSemanticSearch(SemanticSearch):

    def __init__(self, modeL_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(modeL_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = "./cache/chunk_embeddings.npy"
        self.chunk_metadata_path = "./cache/chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        chunks: list[str] = list()
        metadata: list[dict] = list()

        self.documents = documents
        for doc_index, document in enumerate(self.documents):
            doc_id: int = document["id"]
            doc_desc: str = document["description"]

            if len(doc_desc.strip()) == 0:
                continue
            
            self.document_map[doc_id] = document
            desc_chunks = self.semantic_chunk(doc_desc, 4, 1)
            chunks.extend(desc_chunks)

            for chunk_index, _ in enumerate(desc_chunks):
                chunk_dict = dict()
                chunk_dict["movie_idx"] = doc_id
                chunk_dict["chunk_idx"] = chunk_index
                chunk_dict["total_chunks"] = len(desc_chunks)
                metadata.append(chunk_dict)
        
        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = copy.deepcopy(metadata)
        
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(self.chunk_embeddings)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            doc_id: int = doc["id"]
            self.document_map[doc_id] = doc
        
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
            
    def semantic_chunk(self, text: str, max_chunk_size: int, overlap: int):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current_chunk, overlap_chunks = [], [], []

        for sentence in sentences:
            current_chunk.append(sentence)

            if len(current_chunk) == max_chunk_size:
                chunks.append(" ".join(current_chunk))
                overlap_chunks = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_chunks[:]

        if current_chunk and current_chunk != overlap_chunks:
            chunks.append(" ".join(current_chunk))

        return chunks
        
    def search_chunks(self, query: str, limit: int = 10):
        chunk_scores = list()
        
        query_embedding = self.generate_embedding(query)
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_data = dict()
            chunk_data["score"] = similarity_score
            chunk_data["chunk_idx"] = self.chunk_metadata[idx]["chunk_idx"]
            chunk_data["movie_idx"] = self.chunk_metadata[idx]["movie_idx"]
            chunk_scores.append(chunk_data)

        movie_scores = dict()
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            chunk_score = chunk_score["score"]
            if movie_idx not in movie_scores:
                movie_scores[movie_idx] = chunk_score
                continue
            
            current_chunk_score = movie_scores[movie_idx]
            if chunk_score > current_chunk_score:
                movie_scores[movie_idx] = chunk_score
        
        sorted_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        filtered_scores = sorted_scores[:limit]

        movie_results = list()

        for (movie_idx, score) in filtered_scores:
            movie_data = dict()
            movie_data["id"] = movie_idx
            movie_data["title"] = self.document_map[movie_idx]["title"]
            movie_data["description"] = self.document_map[movie_idx]["description"][:100]   
            movie_data["score"] = round(score, 2)
            movie_results.append(movie_data)

        return movie_results

