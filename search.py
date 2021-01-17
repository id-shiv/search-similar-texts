# import required packages
import gensim
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances

class TopicModelling:
    def __init__(self, documents):
        # get list of documents
        self._documents = documents

        # Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
        # Run in terminal: python -m spacy download en
        self._nlp = spacy.load('en', disable=['parser', 'ner'])

        # prepare data for topic modelling
        self._vectorizer, _vectorized = self.__vectorize(self._documents)

        # generate the model
        self._model = self.__model(_vectorized)

        # topic vs words dataframe
        topic_keywords = self.__get_topic_keywords()
        self._df_topic_words = self.__get_topic_words(topic_keywords)

    def __document_to_words(self, documents):
        """Split documents into words

        Args:
            documents ([list()]): [list of documents]
        """
        # deacc=True removes punctuations
        return [gensim.utils.simple_preprocess(str(document), deacc=True) for document in documents]
            
    def __lemmatize(self, words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
        """Lemmatize the input texts

        Args:
            texts ([list]): [list of texts]
            allowed_postags (list, optional): [description]. Defaults to ['NOUN', 'ADJ', 'VERB', 'ADV'].

        Returns:
            [list]: [lemmatized texts]
        """
        texts_out = []
        for document in words:
            doc = self._nlp(" ".join(document)) 
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out

    def __vectorize(self, documents):
        """Prepare data for building LDA model

        Returns:
            [vectorized]: [vectorized texts]
        """
        # print(self._documents[:3])

        # break documents into words
        _words = self.__document_to_words(documents)

        # lemmatize
        _lemmatized = self.__lemmatize(_words, allowed_postags=['NOUN', 'VERB']) # select noun and verb

        # vectorize
        _vectorizer = CountVectorizer(analyzer='word',       
                                        min_df=1,  # minimum reqd occurences of a word 
                                        stop_words='english',  # remove stop words
                                        lowercase=True,  # convert all words to lowercase
                                        token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                        max_features=50000,  # max number of uniq words    
                                    )
        return _vectorizer, _vectorizer.fit_transform(_lemmatized)

    def __model(self, vectorized):
        """Generate LDA model

        Returns:
            [model]: [LDA model object]
        """
        lda_model = LatentDirichletAllocation(n_components=10,  # Number of topics
                                        max_iter=10,  # Max learning iterations
                                        learning_method='online',   
                                        random_state=100,  # Random state
                                        batch_size=32,  # n docs in each learning iter
                                        evaluate_every = -1,  # compute perplexity every n iters, default: Don't
                                        n_jobs = -1,  # Use all available CPUs
                                    )

        lda_model.fit_transform(vectorized)
        return lda_model

    def __get_topic_keywords(self, n_words=20):
        keywords = np.array(self._vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self._model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    def __get_topic_words(self, topic_keywords):
        # Topic - Keywords Dataframe
        df_topic_words = pd.DataFrame(topic_keywords)
        df_topic_words.columns = ['Word '+str(i) for i in range(df_topic_words.shape[1])]
        df_topic_words.index = ['Topic '+str(i) for i in range(df_topic_words.shape[0])]

        return df_topic_words

    def predict(self, text):
        """Predict topic for a given text

        Returns:
            [str]: [predicted topic]
        """
        # Step 1: Clean with simple_preprocess
        _words = self.__document_to_words(text)
        
        # Step 2: Lemmatize
        _lemmatized = self.__lemmatize(_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # Step 3: Vectorize transform
        _vectorized = self._vectorizer.transform(_lemmatized)
        
        # Step 4: LDA Transform
        topic_probability_scores = self._model.transform(_vectorized)
        topic = self._df_topic_words.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
        
        # Step 5: Infer Topic
        infer_topic = self._df_topic_words.iloc[np.argmax(topic_probability_scores), -1]
        
        # topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
        return infer_topic, topic, topic_probability_scores

    def similar(self, text, top_n=5, verbose=False):
        _, _vectorized = self.__vectorize(self._documents)
        lda_model = self._model.transform(_vectorized)

        _, topic, x  = self.predict(text)
        dists = euclidean_distances(x.reshape(1, -1), lda_model)[0]
        doc_ids = np.argsort(dists)[:top_n]
        if verbose:        
            print("Topic KeyWords: ", topic)
            print("Topic Prob Scores of text: ", np.round(x, 1))
            print("Most Similar Doc's Probs:  ", np.round(lda_model[doc_ids], 1))
        return doc_ids, np.take(self._documents, doc_ids)

if __name__ == '__main__':
    # import the data
    texts = list()
    with open('texts.txt', 'r') as f:
        texts = [text.strip() for text in f.readlines() if len(text) > 1]

    data = TopicModelling(documents=texts)

    # Predict the topic
    mytext = ["must be included as part non functional"]
    infer_topic, topic, prob_scores = data.predict(text = mytext)
    # print(topic)
    print(infer_topic)

    # get similar texts
    doc_ids, docs = data.similar(text=mytext, top_n=5)
    print(docs)
