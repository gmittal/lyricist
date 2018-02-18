# lyricist
Experiments with [word2vec](https://en.wikipedia.org/wiki/Word2vec) and [song lyric data](https://www.kaggle.com/mousehead/songlyrics). Unsupervised song recommendation uses semantic vector representations to generate song vectors that can be compared against one another. Song lyric generation uses a character-level LSTM.

### Recommendation
To train the music recommender, run:
```
python train.py
```
To test out the model, run:
```
python evaluate.py
```

### Generation
The lyric generator uses a variant of [Karpathy's char-rnn](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
