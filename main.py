import itertools
import pandas as pd
import re
import tfidf
from util import contractions
from gensim.models import Word2Vec

data = pd.read_csv('./data/songdata.csv') # ~2 million sentences

def preprocess(text):
    try:
        text = contractions.expand(text) if text.index("'") > -1 else text
    except:
        pass
    return re.findall(r'\w+', text.lower())

# Turn lyrics into list of preprocessed sentences
def text2sent(lyrics):
    return list(filter(
        lambda x: len(x) > 0,
        [preprocess(s.rstrip()) for s in lyrics.split('\n')]))

# Turn lyrics into preprocessed word list
def text2doc(lyrics):
    return list(itertools.chain.from_iterable(text2sent(lyrics)))

# Build list of sentences that can be fed to word2vec
def build_sentences(data):
    dataset = []
    for song in data:
        dataset += text2sent(song)
    return dataset

def setup_tfidf():
    for song in data['text']:
        tfidf.init(text2doc(song))

print('Building sentences...')
sentences = build_sentences(data['text'])
setup_tfidf()
print('Training word2vec...')
model = Word2Vec(sentences, workers=4)
model.save('./data/model')
print('Done.')
