import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify embedding model")
    
    embed_parser = subparsers.add_parser("embed_text", help="Create embedding for a given text")
    embed_parser.add_argument("text", type=str, help="The text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embedding model")

    query_embed_parser = subparsers.add_parser("embed_query", help="Create embedding for a given query")
    query_embed_parser.add_argument("query", type=str, help="The query to embed")

    semantic_search_parser = subparsers.add_parser("search", help="Search similar movies")
    semantic_search_parser.add_argument("query", type=str, help="The query to search movies for")
    semantic_search_parser.add_argument("--limit", type=int, default=5, help="the num ")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()