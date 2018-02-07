import math

# frequency of word in document
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

# number of documents containing a word
def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

# inverse document frequency
def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)
