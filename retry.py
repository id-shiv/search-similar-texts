# import required packages
import gensim
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances

# initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
# run in terminal to download: python -m spacy download en
_nlp = spacy.load('en', disable=['parser', 'ner'])

verbose = True

def pprint(message: str, verbose=False):
    if verbose:
        print(message)
        print()

def __documents_to_words(documents):
    """Split documents into words

    Args:
        documents ([list()]): [list of documents]
    """
    # deacc=True removes punctuations
    return [gensim.utils.simple_preprocess(str(document), deacc=True) for document in documents]
          
def _lemmatize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = _nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# import the data
texts = list()
with open('texts.txt', 'r') as f:
    texts = [text.strip() for text in f.readlines() if len(text) > 1]
pprint(f'input texts: {texts[:3]}', False)
pprint(f'number of input texts: {len(texts)}', True)

# split documents into words
words = __documents_to_words(texts)
pprint(f'words in input texts: {words[:3]}', False)

# lemmatize words
lemmatized = _lemmatize(words)
pprint(f'lemmatized texts: {lemmatized[:3]}', False)
_words = [word for document in lemmatized for word in document.split(' ')]
pprint(f'number of words in input texts: {len(_words)}', True)
pprint(f'number of unique words in input texts: {len(set(list(_words)))}', True)

# create document-word matrix
# document as each row, word as each column \ feature
vectorizer = CountVectorizer(   stop_words='english',   
# remove all english stopwords          
                                analyzer='word',     
# If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. 
                                lowercase=True,                   
# convert all words to lowercase
)
vectorized = vectorizer.fit_transform(lemmatized)
pprint(f'number of features(words) in count vectorizer: {len(vectorizer.get_feature_names())}', True)
pprint(f'features in count vectorizer: {vectorizer.get_feature_names()}', False)