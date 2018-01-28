import logging
import pandas as pd
import re
from gensim.models import word2vec

data = pd.read_csv('./data/songdata.csv')

def preprocess(text):
    return ' '.join(re.findall(r'\w+', text.lower()))

def sentences(lyrics):
    return list(filter(
        lambda x: len(x) > 0,
        [preprocess(s.rstrip()) for s in lyrics.split('\n')]))

i = data['song'].index('Dancing Queen')
sent = sentences(data['text'][i])
print(sent)

