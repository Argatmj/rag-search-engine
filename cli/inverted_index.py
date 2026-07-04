import json
import pickle
import string
from nltk.stem import PorterStemmer
from collections import Counter

class InvertedIndex():

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.movies_path = "./data/movies.json"
        self.stop_words_path = "./data/stopwords.txt"
        self.stop_words = None
        self.stemmer = PorterStemmer()
        self.table = str.maketrans("", "", string.punctuation)
        self.term_freq = dict()

        with open(self.stop_words_path) as s:
            self.stop_words = set(s.read().splitlines())

    def __add_document(self, doc_id, text):
        cleaned_text = self.__remove_all_punctuations(text)
        tokens = self.__tokenize_text(cleaned_text)
        shortened_tokens = [token for token in tokens if token not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(token) for token in shortened_tokens]

        for token in stemmed_tokens:
            if doc_id not in self.term_freq:
                self.term_freq[doc_id] = Counter()
            self.term_freq[doc_id][token] += 1
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_tf(self, doc_id, term):
        freq = self.term_freq.get(doc_id[term], 0);
        return freq
        
    def __tokenize_text(self, word: str):
        words = word.lower().split()
        return words

    def __remove_all_punctuations(self, word: str):
        cleaned_word = word.translate(self.table)
        return cleaned_word

    def get_document(self, term):
        doc_ids = self.index[term]
        return doc_ids

    def build(self):
        movies = self.__load_movies()
        
        for i, movie in enumerate(movies):
            print(f"Processing movie {i}")

            movie_id = movie["id"]
            movie_title = movie["title"]
            movie_desc = movie["description"]

            self.docmap[movie_id] = movie
            movie_text = f"{movie_title} {movie_desc}"
            self.__add_document(movie_id, movie_text)

    def __load_movies(self):
        movie_list = []
        with open(self.movies_path) as f:
            movies = json.load(f)
            movie_list = movies["movies"]
        return movie_list

    def save(self):
        with open("./cache/index.pkl", "wb") as i:
            pickle.dump(self.index,i)

        with open("./cache/docmap.pkl", "wb") as d:
            pickle.dump(self.docmap,d)

        with open("./cache/term_freq.pkl", "wb") as t:
            pickle.dump(self.term_freq,t)

    def load(self):
        index, docmap = None, None
        with open("./cache/index.pkl", "rb") as i:
            index = pickle.load(i)
        with open("./cache/docmap.pkl", "rb") as d:
            docmap = pickle.load(d)
        with open("./cache/term_freq.pkl", "rb") as t:
            term_freq = pickle.load(t)

        if index is None or docmap is None or term_freq is None:
            raise FileNotFoundError()
        
        return index, docmap, term_freq

    def build_command(self):
        self.build()
        self.save()