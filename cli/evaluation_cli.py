import argparse
import json
from lib.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("./data/golden_dataset.json") as g:
        golden_dataset = json.load(g)
    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]

    test_cases = golden_dataset["test_cases"]

    for case in test_cases:
        hybird_search = HybridSearch(movie_list)
        query = case["query"]
        relevant_docs = case["relevant_docs"]
        scores = hybird_search.rrf_search(query, 60, limit)

        total_retrieved = len(scores)
        relevant_retrieved_titles = [ rank["document"]["title"] for (doc_id, rank) in scores if rank["document"]["title"] in relevant_docs]
        relevant_retrieved = len(relevant_retrieved_titles)
        total_relevant  = len(relevant_docs)

        recall = relevant_retrieved / total_relevant
        precision = relevant_retrieved / total_retrieved
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        retrieved_titles = ", ".join(relevant_retrieved_titles)
        relevant_titles = ", ".join(relevant_docs)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved : {retrieved_titles}")
        print(f"  - Relevant: {relevant_titles}")

    
if __name__ == "__main__":
    main()