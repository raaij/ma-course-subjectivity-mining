import os
from sklearn.base import TransformerMixin
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path

from sklearn.pipeline import FeatureUnion

PATH_DATA = Path(os.path.dirname(__file__)) / '..' / 'data'


class Text2Embedding(TransformerMixin):
    def __init__(self, embed_source):
        self.embed_source = embed_source
        self.model = None
        if self.embed_source == 'glove':
            path = PATH_DATA / 'glove.twitter.27B.100d.txt'
            w2vfile = PATH_DATA / 'glove.twitter.27B.100d.vec'
            if not Path(w2vfile).is_file():
                glove2word2vec(path, w2vfile)
            self.model = KeyedVectors.load_word2vec_format(w2vfile, binary=False)
        else:
            path = PATH_DATA / 'wiki-news-300d-1M.vec'
            self.model = KeyedVectors.load_word2vec_format(path, binary=False)
        self.n_d = len(self.model['the'])

    def fit_transform(self, X, parameters=[]):
        print('transforming data using customized transformer')
        
        data = []
        for tokenized_tweet in X:
            tokens = tokenized_tweet.split(' ')
            tweet_matrix = np.array([self.model[t] for t in tokens if t in self.model.vocab])
            if len(tweet_matrix) == 0:
                data.append(np.zeros(self.n_d))
            else:
                data.append(np.mean(tweet_matrix, axis=0))
        return np.array(data)

    def transform(self, X):
        return self.fit_transform(X)


# --------------- standard formatters ----------------------

def count_vectorizer(kwargs={}):
    return CountVectorizer(**kwargs)


def tfidf_vectorizer(kwargs={}):
    return TfidfVectorizer(**kwargs)


def text2embeddings(embed_source='glove'):
    return Text2Embedding(embed_source)
