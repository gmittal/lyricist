import itertools
import math
from tqdm import *

_docs = []
word2doc = {}

# frequency of word in document
def tf(word, doc):
    return doc.count(word) / len(doc)

# number of documents containing a word
def n_containing(word):
    return word2doc[word]

# inverse document frequency
def idf(word, docs):
    return math.log(len(docs) / (1 + n_containing(word)))

# Compute tf-idf score of word in a given document
def tfidf(word, doc):
    return tf(word, doc) * idf(word, _docs)

# Add a document to tf-idf model
def add(doc):
    _docs.append(doc)

def init():
    global word2doc
    vocab = list(set(list(itertools.chain.from_iterable(_docs))))
    word2doc = {vocab[i]: sum(1 for doc in _docs if vocab[i] in doc) for i in tqdm(range(0, len(vocab)))}
