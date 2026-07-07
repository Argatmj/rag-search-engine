from lib.hybrid_search import normalize, weighted_search, rrf_search
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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()