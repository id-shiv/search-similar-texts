# Search similar texts using Topic Modelling

Search similar texts given a series of texts by applying Topic Modelling approach.

## Approach

- Assign topics + probabilities to each text.
- Each topic would be associated with words + probabilities.
- Extract topics from new text
- Match the best topic + text based on new text topics.

## Output

- Predict a dominant tag for a given text.
- Find N similar texts for a given text.

## How to run?

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

## Reference

<https://medium.com/@yanlinc/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6>
