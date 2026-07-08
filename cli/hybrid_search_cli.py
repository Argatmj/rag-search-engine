from lib.hybrid_search import normalize, weighted_search, rrf_search
from lib.llm import Model, spell_prompt, rewrite_prompt, enhanced_prompt
import argparse

def normalize_scores(scores: list):
    normalized_scores = normalize(scores)
    if normalized_scores:
        for score in normalized_scores:
            print(f"* {score:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="normalize a given list of numbers")
    normalize_parser.add_argument("scores", type=float, nargs="*")

    weighted_search_parser = subparser.add_parser("weighted-search", help="Use weighted search")
    weighted_search_parser.add_argument("query", type=str, help="The query")
    weighted_search_parser.add_argument("--alpha", type=float, help="The distribution of weights")
    weighted_search_parser.add_argument("--limit", type=int, help="The number of movies shown")

    rrf_search_parser = subparser.add_parser("rrf-search", help="Use weighted search")
    rrf_search_parser.add_argument("query", type=str, help="The query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="The weight in rrf score")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="The number of movies shown")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            query = args.query
            limit = args.limit

            if args.rerank_method:
                limit = limit * 5

            if args.enhance:
                model = Model()
                prompt = None
                
                match args.enhance:
                    case "spell":
                        prompt = spell_prompt(args.query)
                    case "rewrite":
                        prompt = rewrite_prompt(args.query)
                    case "expand":
                        prompt = enhanced_prompt(args.query)
                    case _:
                        raise ValueError(f"{args.enhance} was not an option.")

                enhanced_query = model.get_response(prompt)
                query = enhanced_query[:]
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
        
            rrf_search(query, args.k, args.rerank_method, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()