# import required packages
import gensim

# Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
# Run in terminal: python -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

def sent_to_words(sentences):
    """ Split sentences into words

    Args:
        sentences ([list()]): [list of sentences]
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
    """Lemmatize the input texts

    Args:
        texts ([list]): [list of texts]
        allowed_postags (list, optional): [description]. Defaults to ['NOUN', 'ADJ', 'VERB', 'ADV'].

    Returns:
        [list]: [lemmatized texts]
    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# import the data
texts = list()
with open('texts.txt', 'r') as f:
    texts = [text.strip() for text in f.readlines() if len(text) > 1]

# break documents into words
words = list(sent_to_words(texts))
print(words[:3])

# lemmatize
data_lemmatized = lemmatization(words, allowed_postags=['NOUN', 'VERB']) # select noun and verb
print(data_lemmatized[:3])