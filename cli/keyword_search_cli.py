import argparse
import json
import string
from nltk.stem import PorterStemmer

def search_movies_by_keyword(movies, keyword: str, stop_words: list[str]):
    movie_list = []
    stemmer = PorterStemmer()
    cleaned_keyword = remove_all_punctuations(keyword)
    tokenized_keyword = tokenize(cleaned_keyword)
    shorten_keyword = [token for token in tokenized_keyword if token not in stop_words]
    stemmed_keyword = [stemmer.stem(key) for key in shorten_keyword]

    for movie in movies:
        title = movie["title"]
        cleaned_title = remove_all_punctuations(title)
        titles = cleaned_title.lower()
        tokenized_titles = tokenize(titles)
        shorten_titles = [token for token in tokenized_titles if token not in stop_words]
        stemmed_titles = [stemmer.stem(key) for key in shorten_titles]

        for key in stemmed_keyword:
            if key in stemmed_titles:
                movie_list.append(title)

    return movie_list[:5]

def remove_all_punctuations(word: str):
    table = str.maketrans("", "", string.punctuation)
    cleaned_word = word.translate(table)
    return cleaned_word

def tokenize(word: str):
    words = word.split()
    return words

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            m_file = open("./data/movies.json")
            s_file = open("./data/stopwords.txt")

            movies = json.load(m_file)
            stop_words = s_file.read().splitlines()
            cleaned_stop_words = [remove_all_punctuations(word) for word in stop_words]

            m_list = search_movies_by_keyword(movies["movies"], args.query, cleaned_stop_words)
            for idx, title in enumerate(m_list):
                print(f"{idx+1}. {title}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()