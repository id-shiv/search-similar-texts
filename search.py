# import required packages
import gensim
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV

# initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
# run in terminal to download: python -m spacy download en
_nlp = spacy.load('en', disable=['parser', 'ner'])

verbose = True
n_topics = 10  # number of topics
n_similar_documents = 5  # number of similar topics
n_top_words = 1000  # number of top words per topic
input_text = "data analysis must be considered for evaluating"

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
          
def __lemmatize(documents, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    documents_out = []
    for sent in documents:
        doc = _nlp(" ".join(sent)) 
        documents_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return documents_out

# import the data
documents = list()
with open('texts.txt', 'r') as f:
    documents = [text.strip() for text in f.readlines() if len(text) > 1]
pprint(f'input documents: {documents[:3]}', False)
pprint(f'number of input documents: {len(documents)}', True)

# TODO: add some more text processing if required
# 1. remove punctuations

# split documents into words
words = __documents_to_words(documents)
pprint(f'words in input documents: {words[:3]}', False)

# lemmatize words
lemmatized = __lemmatize(words)
pprint(f'lemmatized documents: {lemmatized[:3]}', False)
_words = [word for document in lemmatized for word in document.split(' ')]
pprint(f'number of words in input documents: {len(_words)}', True)
pprint(f'number of unique words in input documents (before stop word removal): {len(set(list(_words)))}', True)

# vectorize: create document-word matrix
# document as each row, word as each column \ feature
vectorizer = CountVectorizer(   # remove all english stopwords 
                                stop_words='english',   
                                # If a list, that list is assumed to contain stop words, 
                                # all of which will be removed from the resulting tokens. 
                                analyzer='word',     
                                # lowercase all the documents
                                lowercase=True,                   
# convert all words to lowercase
)
vectorized = vectorizer.fit_transform(lemmatized)
pprint(f'number of features(words) in count vectorizer: {len(vectorizer.get_feature_names())}', False)
pprint(f'features in count vectorizer: {vectorizer.get_feature_names()}', False)

#region select the best model parameters using GridSearch
# number of topics 
# learning decay

# Define Search Param
n_test_components = [2, 3, 4, 5, 10, 15, 20, 25, 30]
test_learning_decay = [.5, .7, .9]
search_params = {'n_components': n_test_components, 'learning_decay': test_learning_decay}

# Init the Model
best_lda = LatentDirichletAllocation(   # maximum iterations
                                        max_iter=5,
                                        # Number of topics
                                        n_components=n_topics,
                                        # components_ update method while training, 
                                        # 'online' is faster for large data
                                        learning_method='online',
                                        learning_offset=50,
                                        # number of documents per iteration
                                        batch_size=100,
                                        # Use all available CPUs
                                        n_jobs = -1,
                                        random_state=0
                                )

# init GridSearch class
best_model = GridSearchCV(best_lda, param_grid=search_params)

# perform GridSearch
best_model.fit(vectorized)
GridSearchCV(cv=None, error_score='raise',
                estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                evaluate_every=-1, learning_decay=0.7, learning_method=None,
                learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
                mean_change_tol=0.001, n_components=10, n_jobs=1,
                # n_topics=None, 
                perp_tol=0.1, random_state=None,
                topic_word_prior=None, total_samples=1000000.0, verbose=0),
                # fit_params=None, 
                # iid=True, 
                n_jobs=1,
                param_grid=search_params,
                pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
                scoring=None, verbose=0
            )

# Best Model
model = best_model.best_estimator_
pprint(f"Best Model: {best_model.best_estimator_}", True)
# Model Parameters
pprint(f"Best Model's Params: {best_model.best_params_}", False)
# Log Likelihood Score
pprint(f"Best Log Likelihood Score: {best_model.best_score_}", False)

#endregion

# train the model
model.fit_transform(vectorized)
pprint(f'topic vs word distribution in model: {model.components_}', False)
pprint(f'shape of topic vs word distribution in model: topics - {len(model.components_)} * words - {len(model.components_[0])}', False)
pprint(f'topic vs word distribution in model (normalized): {model.components_ / model.components_.sum(axis=1)[:, np.newaxis]}', False)
pprint(f'number of iterations: {model.n_batch_iter_}', False)

# create Document — Topic matrix
# Columns: Topic0, Topic1, Topic2, ... TopicN - N is number of topics
# Rows: Doc0, Doc1, Doc2, ... DocM - M is number of documents
model_output = model.transform(vectorized)
columns = ["Topic" + str(i) for i in range(model.n_components)]  # column names
indices = ["Doc" + str(i) for i in range(len(documents))]  # index names
# create the doc vs topic matrix with values rounded off to 2 digit decimal
df_document_topic = pd.DataFrame(np.round(model_output, 2), columns=columns, index=indices)
pprint(f'Document vs Topic matrix shape: {df_document_topic.shape} - (number of documents, number of topics)', False)

# get dominant topic for each document
# Dominant Topic: Topic with maximum weight for a given document
dominant_topic = np.argmax(df_document_topic.values, axis=1)  # indices of word with maximum weight per topic
# create a new column to note the dominant word for every topic
df_document_topic['dominant_topic'] = dominant_topic
pprint(f'Document vs Topic matrix with index of dominant topic:\n {df_document_topic}', False)

# create Topic - Word matrix
# Columns: Actual words in ascending order
# Rows: Topic0, Topic1, Topic2, ... TopicN - N is number of topics
df_topic_word = pd.DataFrame(model.components_)  # model.components_ = topic vs word distribution

# assign Column and Index
df_topic_word.columns = vectorizer.get_feature_names()  # words
df_topic_word.index = columns  # topics
pprint(f'Topic Vs Word distribution (with words as actual column names):\n {df_topic_word}', False)
pprint(f'Shape of Topic Vs Word distribution (with words as actual column names): {df_topic_word.shape} - (number of topics, number of words)', False)

# Create Topic - Keyword (top N words per topic) matrix

# get top N words for each topic
all_words = np.array(vectorizer.get_feature_names())
topic_keywords = []
for topic_weights in model.components_:  # for every topic
    # Convert to negative weights 
    # sort in ascending order
    # fetch indices of only the top N words
    # [144 238 227 99 180 ...]
    top_keyword_locs = (-topic_weights).argsort()[:n_top_words]
    # Get the actual top N words for the topic using the indices
    topic_keywords.append(all_words.take(top_keyword_locs))

pprint(f'Top N words for every topic:\n {topic_keywords}', False)
pprint(f'Shape of top N words for every topic:\nNumber of topics: {len(topic_keywords)}\nNumber of top words in topic: {len(topic_keywords[0])}', True)

# Topic - Keywords Dataframe
# Columns: Word0, Word1, Word2, ... WordN - N : Top N words
# Rows: Topic0, Topic1, Topic2, ... TopicM - M : Number fo topics
# Word0 has the height weightage and is the dominant word in the topic. i.e. topic name
df_topic_keywords = pd.DataFrame(topic_keywords)

# assign Column and Index
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]  # columns = topn N words
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]  # rows = topics
pprint(f'Topic Vs Top N Words:\n {df_topic_keywords}', False)
pprint(f'Shape of Topic Vs Top N Words : {df_topic_keywords.shape} - (number of topics, number of top N words)', False)

# Predict a topic for given input text

# Step 1: Clean with simple_preprocess
input_words = __documents_to_words([input_text])

# Step 2: Lemmatize
input_lemmatized = __lemmatize(input_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Step 3: Vectorize transform
input_vectorized = vectorizer.transform(input_lemmatized)

# Step 4: LDA Transform
input_probability_scores = model.transform(input_vectorized)

# Step 5: Infer Topic
# get the top N words associated with the topic
topics = df_topic_keywords.iloc[np.argmax(input_probability_scores), : n_top_words].values.tolist()
# get the dominant word in the topic (i.e first element - index 0, since words are sorted with weights in ascending order)
infer_topic = df_topic_keywords.iloc[np.argmax(input_probability_scores), 0]

pprint(f'Input text: {input_text}', True)
pprint(f'Probability scores for each topic against input text:\n{input_probability_scores}', False)
pprint(f'Top N words in predicted topic:\n{topics}', False)
pprint(f'Predicted topic (dominant word in the topic): {infer_topic}', True)

# Get similar documents for an input text
# Get distance between input text and all documents
input_text_documents_dist = euclidean_distances(input_probability_scores.reshape(1, -1), model_output)[0]
pprint(f'Euclidean Distances of input text Vs N documents: \n{input_text_documents_dist}', False)

# sort the distances and pick the top N document indices
document_indices = np.argsort(input_text_documents_dist)[:n_similar_documents]

# get the similar document by indices
similar_documents = np.take(documents, document_indices)
pprint(f'Top N similar documents: \n{similar_documents}', True)
pprint(f'Top matching document: {documents[document_indices[0]]}', True)
