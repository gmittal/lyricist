import pandas as pd
import re
from gensim.models import word2vec

data = pd.read_csv('./data/songdata.csv')
print data.head()

