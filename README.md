# Search similar texts using Topic Modelling

For a given text, retrieve the associated topic and top N similar texts by Topic Modelling approach (LDA).

## Approach

* For a given set of documents:
  * Find the ideal model parameters for topic modelling (LDA) i.e. number of topics, learning decay.
  * Generate document-word matrix with weightage of each word.
  * Generate topic-word matrix with number of words limited to each topic.

* Predict:
  * For a given text, retreive the best topic.
  * Get the dominant word in the predicted topic.
  * Dominant word ultimately is the topic tag

* Get similar douments:
  * For a given text, derive distance with all documents.
  * Get the top N documents based on distance.

## Output

* Predict a topic (dominant word associated to the topic) for a given text.
* Find N similar texts for a given text and documents.

## Reference

<https://medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6>
