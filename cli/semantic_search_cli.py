import argparse
import re
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search
from lib.chunk_semantic_search import embed_chunks, search_chunked

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

    chunk_parser = subparsers.add_parser("chunk", help="Splitting text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="The text to be split")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="The size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, help="The overlap of each chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Splitting text into smaller pieces with semantics")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to be split")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="The size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="The overlap of each chunk")

    subparsers.add_parser("embed_chunks", help="Create embeddings for each chunk for the whole movie documents" )

    search_chunk_parser = subparsers.add_parser("search_chunked", help="Search movies by semantic chunking ")
    search_chunk_parser.add_argument("query", type=str, help="The query to search a movie")
    search_chunk_parser.add_argument("--limit", type=int, default=5, help="The number of top N movies")

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
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()

def chunk(text: str, chunk_size: int, overlap: int):
    chunks = list()
    total_char, current_chunk = len(text), []
    words = text.split()

    for word in words:
        current_chunk.append(word)

        if len(current_chunk) == chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_chunks = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_chunks[:]
            
    if current_chunk and current_chunk != overlap_chunks:
        chunks.append(" ".join(current_chunk))

    print(f"Chunking {total_char} characters..")
    for index, chunk in enumerate(chunks):
        print(f"{index+1}. {chunk}")

def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    cleaned_text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    chunks, current_chunk, overlap_chunks = [], [], []
    char_length = len(cleaned_text)

    for sentence in sentences:
        current_chunk.append(sentence)

        if len(current_chunk) == max_chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_chunks = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_chunks[:]

    if current_chunk and current_chunk != overlap_chunks:
        chunks.append(" ".join(current_chunk))
    
    print(f'Semantically chunking {char_length} characters..')
    for index, chunk in enumerate(chunks):
        print(f"{index+1}. {chunk}")

if __name__ == "__main__":
    main()