import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.llm import Model, response_prompt, summarize_prompt, citate_prompt, question_prompt

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser("summarize", help="Summarize the results of the query")
    summarize_parser.add_argument("query", type=str, help="The query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Top N results")

    citations_parser = subparsers.add_parser("citations", help="Provide the results of the query with citations")
    citations_parser.add_argument("query", type=str, help="The query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Top N results")

    question_parser = subparsers.add_parser("question", help="Provide answer based on a question")
    question_parser.add_argument("query", type=str, help="The question")
    question_parser.add_argument("--limit", type=int, default=5, help="Top N results")

    args = parser.parse_args()

    query = args.query
    with open("./data/movies.json") as f:
        movies = json.load(f)
        movie_list = movies["movies"]
    hybird_search = HybridSearch(movie_list)
    model = Model()

    match args.command:
        case "rag":
            scores = hybird_search.rrf_search(query, 60, 5)
            prompt = response_prompt(query, scores)
            response = model.get_response(prompt)

            print(f"Search Results: ")
            for (doc_id, rank) in scores:
                print(f"- {rank['document']['title']}")

            print(f"RAG Response:")
            print(response)
        case "summarize":
            scores = hybird_search.rrf_search(query, 60, args.limit)
            prompt = summarize_prompt(query, scores)
            response = model.get_response(prompt)

            print(f"Search Results: ")
            for (doc_id, rank) in scores:
                print(f"- {rank['document']['title']}")

            print(f"LLM summary:")
            print(response)
        case "citations":
            scores = hybird_search.rrf_search(query, 60, args.limit)
            prompt = citate_prompt(query, scores)
            response = model.get_response(prompt)

            print(f"Search Results: ")
            for (doc_id, rank) in scores:
                print(f"- {rank['document']['title']}")

            print(f"LLM Answer:")
            print(response)
        case "question":
            scores = hybird_search.rrf_search(query, 60, args.limit)
            prompt = question_prompt(query, scores)
            response = model.get_response(prompt)

            print(f"Search Results: ")
            for (doc_id, rank) in scores:
                print(f"- {rank['document']['title']}")

            print(f"LLM Answer:")
            print(response)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()