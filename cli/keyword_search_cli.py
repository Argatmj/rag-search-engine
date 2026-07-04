import argparse
import json
import string
import math
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

def search_movies_by_keyword(keyword: str, stop_words: list[str], index: dict):
    movies_ids = list()
    stemmed_keyword = tokenizer(keyword, stop_words)

    for keyword in stemmed_keyword:
        ids = index.get(keyword)
        if ids is not None:
            for idx in ids:
                movies_ids.append(idx);
                if len(movies_ids) >= 5:
                    break
        if len(movies_ids) >= 5:
            break

    return movies_ids

def remove_all_punctuations(word: str):
    table = str.maketrans("", "", string.punctuation)
    cleaned_word = word.translate(table)
    return cleaned_word

def tokenize(word: str):
    words = word.split()
    return words

def tokenizer(term: str, stop_words):
    stemmer = PorterStemmer()
    cleaned_term = remove_all_punctuations(term)
    tokenized_term = tokenize(cleaned_term)
    shorten_term = [token for token in tokenized_term if token not in stop_words]
    stemmed_term = [stemmer.stem(key) for key in shorten_term]

    return stemmed_term


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="build inverted index and saves it")
    
    tf_parser = subparsers.add_parser("tf", help="Get the number of frequencies for a given term")
    tf_parser.add_argument("document_id", type=int, help= "the document id look at")
    tf_parser.add_argument("term", type=str, help="The term to look tf for")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency of a given term")
    idf_parser.add_argument("term", type=str, help="The term to look idf for ")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score of a given term and document id")
    tfidf_parser.add_argument("document_id", type=int, help="The document id to look for tf")
    tfidf_parser.add_argument("term", type=str, help="The term to look tf-idf for")


    args = parser.parse_args()
    inverted_index = InvertedIndex()

    try:
        index, docmap, term_freq = inverted_index.load()
    except FileNotFoundError:
        print("File not found")
        print("Rebuilding cache...")
        inverted_index.build_command()
        exit()

    s_file = open("./data/stopwords.txt")
    stop_words = s_file.read().splitlines()
    cleaned_stop_words = [remove_all_punctuations(word) for word in stop_words]

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            m_file = open("./data/movies.json")

            movies = json.load(m_file)

            movies_id = search_movies_by_keyword(args.query, cleaned_stop_words, index)
            for ids in movies_id:
                movies = docmap[ids]
                print(f"{ids}: {movies['title']}")

        case "build":
            inverted_index.build_command()
        case "tf":
            doc_id = args.document_id
            term = args.term
            term = tokenizer(term, cleaned_stop_words)
            freq = 0

            if len(term) == 1:
                freq = term_freq[doc_id][term[0]]
            
            print(f"Frequency of `{term[0]}` in document {doc_id} is {freq}")
        case "idf":
            term = args.term
            tokenized_term = tokenizer(term, cleaned_stop_words)[0]
            doc_count = len(docmap.keys())
            match_count = len(index.get(tokenized_term, []))
            idf = math.log((doc_count + 1) / (match_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            doc_id = args.document_id
            term = tokenizer(args.term, cleaned_stop_words)
            tf = 0
            if len(term) == 1:
                tf = term_freq[doc_id][term[0]]

            doc_count = len(docmap.keys())
            match_count = len(index.get(term[0],0))
            idf = math.log((doc_count + 1) / (match_count + 1))

            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{doc_id}': {tf_idf:.2f}")

            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()