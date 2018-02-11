from util import contractions
import itertools
import pandas
import pickle
import re
import tfidf
from tqdm import *
from gensim.models import Word2Vec

data = pandas.read_csv('./data/songdata.csv') # ~2 million sentences

def preprocess(_text):
    try:
        _text = contractions.expand(_text) if _text.index("'") > -1 else _text
    except:
        pass
    return re.findall(r'\w+', _text.lower())

# Turn lyrics into list of preprocessed sentences
def text2sent(_lyrics):
    return list(filter(
        lambda x: len(x) > 0,
        [preprocess(s.rstrip()) for s in _lyrics.split('\n')]))

# Turn lyrics into preprocessed word list
def text2doc(_lyrics):
    return list(itertools.chain.from_iterable(text2sent(_lyrics)))

# Build list of sentences that can be fed to word2vec
def build_sentences(_data):
    dataset = []
    for song in _data:
        dataset += text2sent(song)
        tfidf.add(text2doc(song))
    return dataset

# Turn lyrics into a document vector
def doc2vec(_lyrics, _model):
    processed = text2doc(_lyrics)
    return sum(map(lambda z: tfidf.tfidf(z, processed) * _model[z], processed))

print('Building sentences...')
sentences = build_sentences(data['text'][:1000])
tfidf.init()

print('Training word2vec...')
model = Word2Vec(sentences, min_count=1, workers=4)
model.save('./data/model')

print('Building song vectors...')
vec = {data['song'][i]: doc2vec(data['text'][i], model) for i in tqdm(range(0, len(data['text'][:1000])))}
with open('./data/songvec.pickle', 'wb') as handle:
    pickle.dump(vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')
