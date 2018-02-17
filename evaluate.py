from gensim.models import Word2Vec
import tfidf
import pickle

model = Word2Vec.load('./data/model')
# print(model.most_similar('wiz', topn=1))

with open('./data/songvec.pickle', 'rb') as handle:
    songs = pickle.load(handle)

print(songs['Dancing Queen'])
