import argparse
from lib.multimodal_search import verify_image_embedding, image_search

def main():
    parser = argparse.ArgumentParser(description="Multimodal search")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify_image_embedding", help="Verify image embed")
    verify_parser.add_argument("image", type=str, help="Path to image")

    image_search_parser = subparser.add_parser("image_search", help="Search movie with a given image")
    image_search_parser.add_argument("image", type=str, help="Path to image")
  
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            image_search(args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()