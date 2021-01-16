# import the data
texts = list()
with open('texts.txt', 'r') as f:
    texts = [text.strip() for text in f.readlines() if len(text) > 1]

print(texts)