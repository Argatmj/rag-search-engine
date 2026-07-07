import os 
import json
from lib.chunk_semantic_search import ChunkSemanticSearch
from lib.inverted_index import InvertedIndex

def normalize(scores: list):
    length = len(scores)
    if length == 0:
       return None
    
    normalized_scores = []
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score: 
        normalized_scores = [1] * length
    else:
        for _ , score in enumerate(scores):
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_scores.append(normalized_score)
    
    return normalized_scores

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def weighted_search(query: str, alpha: float, limit: int = 5):

    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]
    hybird_search = HybridSearch(movie_list)
    scores = hybird_search.weighted_search(query, alpha, limit)

    for idx, (_, score) in enumerate(scores):
        print(f"{idx+1}. {score['document']['title']}")
        print(f"  Hybrid_score: {score['hybrid_score']:.3f}")
        print(f"  BM25: {score['keyword_score']:.3f}, Semantic: {score['semantic_score']:.3f}")
        print(f"  {score['document']['description'][:30]}...")

def rrf_search(query: str, k: int, limit: int = 10):
    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]
    hybird_search = HybridSearch(movie_list)
    scores = hybird_search.rrf_search(query, k, limit)

    for idx, (_, rank) in enumerate(scores):
        print(f"{idx+1}. {rank['document']['title']}")
        print(f"  RRF Score: {rank['rrf_score']:.3f}")
        print(f"  BM25 Rank: {rank['bm25_rank']}, Semantic Rank: {rank['semantic_rank']}")
        print(f"  {rank['document']['description'][:30]}...")


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_score_list = self._bm25_search(query, 500*limit)
        semantic_score_list = self.semantic_search.search_chunks(query, 500*limit)

        bm25_scores = [ (score) for (doc_id, score) in bm25_score_list]
        semantic_scores = [ movie["score"] for movie in semantic_score_list ]

        normalized_bm25_scores = normalize(bm25_scores)
        normalized_semantic_scores = normalize(semantic_scores)

        scores = dict()

        for i, (doc_id, _) in enumerate(bm25_score_list):
            scores[doc_id] = {
                "keyword_score": normalized_bm25_scores[i],
                "document": self.idx.docmap[doc_id],
                "semantic_score": 0
            }

        for i, item in enumerate(semantic_score_list):
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = {
                    "keyword_score": 0,
                    "document": self.idx.docmap[doc_id]
                }

            scores[doc_id]["semantic_score"] = normalized_semantic_scores[i]
        
        for doc_id, score in scores.items():
            key_score = score["keyword_score"]
            semantic_score = score["semantic_score"]
            hy_score = hybrid_score(key_score, semantic_score, alpha)
            scores[doc_id]["hybrid_score"] = hy_score

        sorted_scores = sorted(scores.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)
        return sorted_scores[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_score_list = self._bm25_search(query, 500*limit)
        semantic_score_list = self.semantic_search.search_chunks(query, 500*limit)

        ranks = dict()

        for i, (doc_id, _) in enumerate(bm25_score_list):
            rank = i + 1
            ranks[doc_id] = {
                "rrf_score": rrf_score(rank, k),
                "document": self.idx.docmap[doc_id],
                "bm25_rank": rank,
                "semantic_rank": (500*limit) + 1
            }

        for i, movie in enumerate(semantic_score_list):
            rank = i + 1
            doc_id = movie["id"]
            score = rrf_score(rank, k)
            if doc_id in ranks:
                ranks[doc_id]["rrf_score"] += score
            else:
                ranks[doc_id] = {
                    "rrf_score": score,
                    "document": self.idx.docmap[doc_id],
                    "bm25_rank": (500*limit) + 1
                }
            ranks[doc_id]["semantic_rank"] = rank

        sorted_rank = sorted(ranks.items(), key= lambda x: x[1]["rrf_score"], reverse=True)
        return sorted_rank[:limit]     