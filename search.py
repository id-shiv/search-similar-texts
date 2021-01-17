# import required packages
import gensim
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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

#region prepare data

# break documents into words
words = list(sent_to_words(texts))
# print(words[:3])

# lemmatize
data_lemmatized = lemmatization(words, allowed_postags=['NOUN', 'VERB']) # select noun and verb
# print(data_lemmatized[:3])

# vectorize
vectorizer = CountVectorizer(analyzer='word',       
                                min_df=2,  # minimum reqd occurences of a word 
                                stop_words='english',  # remove stop words
                                lowercase=True,  # convert all words to lowercase
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                max_features=50000,  # max number of uniq words    
                            )
data_vectorized = vectorizer.fit_transform(data_lemmatized)
# print(data_lemmatized)

#endregion

# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=10,  # Number of topics
                                        max_iter=10,  # Max learning iterations
                                        learning_method='online',   
                                        random_state=100,  # Random state
                                        batch_size=32,  # n docs in each learning iter
                                        evaluate_every = -1,  # compute perplexity every n iters, default: Don't
                                        n_jobs = -1,  # Use all available CPUs
                                    )
lda_output = lda_model.fit_transform(data_vectorized)
# print(lda_model)

# predict
# Define function to predict topic for a given text document.
def predict_topic(text, df_topic_keywords, nlp=nlp):
    global sent_to_words
    global lemmatization
    global vectorizer
    global lda_model
    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))
    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)
    ## Step 4: LDA Transform
    topic_probability_scores = lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
    
    # Step 5: Infer Topic
    infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
    
    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
    return infer_topic, topic, topic_probability_scores

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

# Predict the topic
mytext = ["testing must be included as part non functional"]
infer_topic, topic, prob_scores = predict_topic(text = mytext, df_topic_keywords=df_topic_keywords)
print(topic)
print(infer_topic)

# assign domninant topic to all documents
def apply_predict_topic(text, df_topic_keywords):
    text = [text]
    infer_topic, topic, prob_scores = predict_topic(text = text, df_topic_keywords=df_topic_keywords)
    return(infer_topic)
tagged_texts = [{document: apply_predict_topic(document, df_topic_keywords)} for document in texts]

# get similar texts
from sklearn.metrics.pairwise import euclidean_distances

def similar_documents(text, doc_topic_probs, documents = texts, nlp=nlp, top_n=5, verbose=False, df_topic_keywords=df_topic_keywords):
    _, topic, x  = predict_topic(text, df_topic_keywords)
    dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    if verbose:        
        print("Topic KeyWords: ", topic)
        print("Topic Prob Scores of text: ", np.round(x, 1))
        print("Most Similar Doc's Probs:  ", np.round(doc_topic_probs[doc_ids], 1))
    return doc_ids, np.take(documents, doc_ids)

doc_ids, docs = similar_documents(text=mytext, doc_topic_probs=lda_output, documents = texts, top_n=1, verbose=True)
print('\n', docs[0][:500])
print()