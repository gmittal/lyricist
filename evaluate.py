import numpy as np
import operator
import pickle
import tfidf
from gensim.models import Word2Vec

model = Word2Vec.load('./data/model')
with open('./data/songvec.pickle', 'rb') as handle:
    songs = pickle.load(handle)

def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def nn(v, s):
    p = {song: cos_sim(v, vec) for song, vec in s.items()}
    return sorted(p.items(), key=operator.itemgetter(1), reverse=True)

def recommend_songs(song, top_n=10):
    return nn(songs[song], songs)[:top_n]

print(recommend_songs('Dancing Queen'))
