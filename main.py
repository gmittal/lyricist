import itertools
import pandas as pd
import re
from util import contractions
from gensim.models import Word2Vec

data = pd.read_csv('./data/songdata.csv') # ~2 million sentences

def preprocess(text):
    try:
        text = contractions.expand(text) if text.index("'") > -1 else text
    except:
        pass
    return re.findall(r'\w+', text.lower())

def text2Sent(lyrics):
    return list(filter(
        lambda x: len(x) > 0,
        [preprocess(s.rstrip()) for s in lyrics.split('\n')]))

def buildSentences():
    dataset = []
    for song in data['text']:
        dataset += text2Sent(song)
    return dataset

print('Building sentences...')
sentences = buildSentences()
print('Training word2vec...')
model = Word2Vec(sentences, workers=4)
model.save('./data/model')
