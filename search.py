# import required packages
import gensim
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances

class TopicModelling:
    """Class for topic modelling, predict a dominant tag and find n similar texts for a given text

    Input:
    documents - list of texts
    """
    def __init__(self, documents):
        # get list of documents
        self._documents = documents

        # Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
        # Run in terminal to download: python -m spacy download en
        self._nlp = spacy.load('en', disable=['parser', 'ner'])

        # prepare data for topic modelling
        self._vectorizer, _vectorized = self.__vectorize(self._documents)

        # generate the model
        self._model = self.__model(_vectorized)

        # topic vs words dataframe
        topic_keywords = self.__get_topics_keywords()
        self.df_topic_words = self.__get_topics_words(topic_keywords)
        print(self.df_topic_words)

    def __documents_to_words(self, documents):
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
        _words = self.__documents_to_words(documents)

        # lemmatize
        _lemmatized = self.__lemmatize(_words, allowed_postags=['NOUN', 'VERB']) # select noun and verb

        # vectorize
        _vectorizer = CountVectorizer(analyzer='word',       
                                        # min_df=3,  # minimum reqd occurences of a word in documents
                                        stop_words='english',  # remove stop words
                                        lowercase=True,  # convert all words to lowercase
                                        # token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                        # max_features=50000,  # max number of uniq words    
                                    )
        
        return _vectorizer, _vectorizer.fit_transform(_lemmatized)

    def __model(self, vectorized, n_topics=2):
        """Generate LDA model

        Returns:
            [model]: [LDA model object]
        """
        lda_model = LatentDirichletAllocation(n_components=n_topics,  # Number of topics
                                        max_iter=10,  # Max learning iterations
                                        learning_method='online',   
                                        random_state=100,  # Random state
                                        batch_size=256,  # n docs in each learning iter
                                        evaluate_every = -1,  # compute perplexity every n iters, default: Don't
                                        n_jobs = -1,  # Use all available CPUs
                                    )

        lda_model.fit_transform(vectorized)
        return lda_model

    def __get_topics_keywords(self, n_words=100):
        """Get N keywords associated to all topics

        Args:
            n_words (int, optional): [description]. Defaults to 20.

        Returns:
            [list()]: [topic vs keywords]
        """
        keywords = np.array(self._vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self._model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    def __get_topics_words(self, topic_keywords):
        """Get words associated to all topics

        Args:
            topic_keywords ([list()]): [topics vs keywords]

        Returns:
            [pandas.DataFrame()]: [Pandas DataFrame of topics vs words]
        """
        # Topic - Keywords Dataframe
        df_topic_words = pd.DataFrame(topic_keywords)
        df_topic_words.columns = ['Word '+str(i) for i in range(df_topic_words.shape[1])]
        df_topic_words.index = ['Topic '+str(i) for i in range(df_topic_words.shape[0])]

        return df_topic_words

    def predict(self, document, verbose: False):
        """Predict dominant tags for input document

        Args:
            document (str): document

        Returns:
            tuple: 
                ID of predicted topic,
                Words in predicted topic,
                Dominant word in predicted topic,
                Probability scores of predicted topic
        """
        # Step 1: Clean with simple_preprocess
        _words = self.__documents_to_words([document])
        
        # Step 2: Lemmatize
        _lemmatized = self.__lemmatize(_words, allowed_postags=['NOUN', 'VERB'])
        # _lemmatized = self.__lemmatize(_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # print(_lemmatized)

        # Step 3: Vectorize transform
        _vectorized = self._vectorizer.transform(_lemmatized)
        # print(_vectorized)

        # Step 4: LDA Transform
        _topic_probability_scores = self._model.transform(_vectorized)
        # print(_topic_probability_scores)
        
        # Step 5: Infer Topic
        # get the top 20 matching topic words
        _topic_id = np.argmax(_topic_probability_scores)
        _words_in_topic = self.df_topic_words.iloc[_topic_id].values.tolist()
        # infer dominant topic with maximum probability (last word has maximum probability)
        _dominant_word_in_topic = self.df_topic_words.iloc[_topic_id, -1]
        
        if verbose:
            print(f'Probability scores of topics: {_topic_probability_scores}')
            print(f'ID of predicted topic: {_topic_id}')
            print(f'Words in predicted topic: {_words_in_topic}')
            print(f'Dominant word in predicted topic: {_dominant_word_in_topic}')
        # topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
        return _topic_id, _words_in_topic, _dominant_word_in_topic, _topic_probability_scores

    def similar(self, document, top_n=5, verbose=False):
        """Get top N similar documents for a given document

        Args:
            document (str): document
            top_n (int, optional): [top N similar documents]. Defaults to 5.
            verbose (bool, optional): [for additional console logging]. Defaults to False.

        Returns:
            [list(), list()]: [list of document IDs, list of documents] for top N similar documents
        """
        _, _vectorized = self.__vectorize(self._documents)
        lda_model = self._model.transform(_vectorized)

        _topic_id, _words_in_topic, _dominant_word_in_topic, _topic_probability_scores  = self.predict(document, verbose)
        dists = euclidean_distances(_topic_probability_scores.reshape(1, -1), lda_model)[0]
        doc_ids = np.argsort(dists)[:top_n]
        if verbose:        
            # print("Topic KeyWords: ", _words_in_topic)
            # print("Topic Prob Scores of text: ", np.round(_topic_probability_scores, 1))
            # print("Most Similar Doc's Probs:  ", np.round(lda_model[doc_ids], 1))
            print(f'Similar document IDs: {doc_ids}')
            print(f'Similar documents: {np.take(self._documents, doc_ids)}')
        return doc_ids, np.take(self._documents, doc_ids)

if __name__ == '__main__':
    # import the data
    texts = list()
    with open('texts.txt', 'r') as f:
        texts = [text.strip() for text in f.readlines() if len(text) > 1]

    # process input texts and generate model
    tm = TopicModelling(documents=texts)


    # Predict the topic
    input_text = "increase participants in testing and data analysis"
    print(f'\nInput Text: {input_text}\n')
    print('*'*50)
    print('Prediction\n')
    _topic_id, _words_in_topic, _dominant_word_in_topic, _topic_probability_scores = tm.predict(document=input_text, verbose=True)
    print(len(_words_in_topic))
    # get similar texts
    print()
    print('*'*50)
    print('Similar texts\n')
    similar_doc_ids, similar_documents = tm.similar(document=input_text, top_n=5, verbose=True)
