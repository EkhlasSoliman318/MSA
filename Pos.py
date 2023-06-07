import nltk
from collections import Counter
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from config import *


with open("file.txt", "r") as f:
      corpus  =  f.read().split(',')
print(corpus)
tokens, tags , counts , results = [],[], [],[]
for sent in corpus:
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    counts = Counter( tag for word,  tag in tags)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp  =nltk.RegexpParser(grammar)
    result = cp.parse(tags)
    results.append(result)

# print(tokens)
# print(tags)
# print(counts)
print(results)
with open("POS.txt", "w") as output:
    for item in results:
        output.write(str(item))