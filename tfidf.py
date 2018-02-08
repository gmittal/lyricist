import math

_docs = []

# frequency of word in document
def tf(word, doc):
    return doc.count(word) / len(doc)

# number of documents containing a word
def n_containing(word, docs):
    return sum(1 for doc in _docs if word in doc)

# inverse document frequency
def idf(word, docs):
    return math.log(len(docs) / (1 + n_containing(word, docs)))

# Compute tf-idf score of word in a given document
def tfidf(word, doc):
    return tf(word, doc) * idf(word, _docs)

# Add a document to tf-idf model
def init(doc):
    _docs.append(doc)
