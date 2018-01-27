import pandas as pd
import re
from gensim.models import word2vec

data = pd.read_csv('./data/songdata.csv')

def preprocess(text):
    pass

print(len(data['text']))

