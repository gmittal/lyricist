from gensim.models import Word2Vec
import tfidf

model = Word2Vec.load('./data/model')
print(model.most_similar('wiz', topn=1))
