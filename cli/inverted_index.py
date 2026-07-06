import os
import json
import math
import pickle
import string
from nltk.stem import PorterStemmer
from collections import Counter

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex():

    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.doc_lengths = dict()
        self.movies_path = "./data/movies.json"
        self.stop_words_path = "./data/stopwords.txt"
        self.stop_words = None
        self.stemmer = PorterStemmer()
        self.table = str.maketrans("", "", string.punctuation)
        self.term_freq = dict()

        self.s_file = open("./data/stopwords.txt")
        self.stop_words = self.s_file.read().splitlines()

        with open(self.stop_words_path) as s:
            self.stop_words = set(s.read().splitlines())

        self.doc_lengths_path = os.path.join("./cache/", "doc_lengths.pkl")

    def bm25_search(self, query, limit: int = 5):
        tokens = self.__tokenizer(query)
        bm25_scores = dict()

        for doc_id in self.docmap.keys():
            bm25_score_sum = 0
            for token in tokens:
                bm25_score = self.bm25(doc_id, token)
                bm25_score_sum += bm25_score
            bm25_scores[doc_id] = bm25_score_sum

        sorted_bm25_score_list = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_bm25_score_list[:limit]
        
    def bm25(self, doc_id, term) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        bm25_score = bm25_idf * bm25_tf
        return bm25_score
    
    def __add_document(self, doc_id, text):
        cleaned_text = self.__remove_all_punctuations(text)
        tokens = self.__tokenize_text(cleaned_text)
        shortened_tokens = [token for token in tokens if token not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(token) for token in shortened_tokens]
        
        token_count = len(stemmed_tokens)
        self.doc_lengths[doc_id] = token_count

        for token in stemmed_tokens:
            if doc_id not in self.term_freq:
                self.term_freq[doc_id] = Counter()
            self.term_freq[doc_id][token] += 1
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_tf(self, doc_id, term):
        freq = self.term_freq.get(doc_id, {}).get(term, 0)
        return freq
    
    def get_bm25_idf(self, term: str) -> float:
        doc_count = len(self.docmap.keys())
        match_count = len(self.index.get(term,[]))
        bm25_idf = math.log((doc_count - match_count + 0.5) / (match_count + 0.5) + 1)
        
        return bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: int = BM25_K1, b: int = BM25_B): 
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def __tokenize_text(self, word: str):
        words = word.lower().split()
        return words
    
    def __tokenizer(self, term: str):
        stop_words = [self.__remove_all_punctuations(word) for word in self.stop_words]
        cleaned_term = self.__remove_all_punctuations(term)
        tokenized_term = self.__tokenize_text(cleaned_term)
        shorten_term = [token for token in tokenized_term if token not in stop_words]
        stemmed_term = [self.stemmer.stem(key) for key in shorten_term]

        return stemmed_term

    def __remove_all_punctuations(self, word: str):
        cleaned_word = word.translate(self.table)
        return cleaned_word
    
    def __get_avg_doc_length(self) -> float:
        sum_length = 0 

        for length in self.doc_lengths.values():
            sum_length += length
        
        if sum_length == 0: 
            return 0.0
        
        return sum_length / len(self.doc_lengths.keys())

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

        with open(self.doc_lengths_path, "wb") as l:
            pickle.dump(self.doc_lengths,l)

    def load(self):
        index, docmap = None, None
        with open("./cache/index.pkl", "rb") as i:
            index = pickle.load(i)
        with open("./cache/docmap.pkl", "rb") as d:
            docmap = pickle.load(d)
        with open("./cache/term_freq.pkl", "rb") as t:
            term_freq = pickle.load(t)
        with open(self.doc_lengths_path, "rb") as l:
            doc_lengths = pickle.load(l)

        if index is None or docmap is None or term_freq is None or doc_lengths is None:
            raise FileNotFoundError()
        
        self.index = index
        self.docmap = docmap
        self.term_freq = term_freq
        self.doc_lengths = doc_lengths
        
        return index, docmap, term_freq, doc_lengths

    def build_command(self):
        self.build()
        self.save()