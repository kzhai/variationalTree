# Import the corpus and functions used from nltk library
#from nltk.corpus import reuters
#from nltk.corpus import genesis
from nltk.probability import LidstoneProbDist
from nltk.model import NgramModel

from nltk.corpus import brown

#m = NgramModel(1, [str(i) for i in [1,2,3,4,5]])
#print m.prob('1', [])

# Tokens contains the words for Genesis and Reuters Trade
tokens = set(brown.words())
words = [];

for word in tokens:
    words.extend([char for char in word.lower()]);
    words.extend(['\t']);

#tokens = list(genesis.words('english-kjv.txt'))
#tokens.extend(list(reuters.words(categories = 'trade')))

# estimator for smoothing the N-gram model
estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.02)

# N-gram language model with 3-grams
#model = NgramModel(3, tokens, estimator)
model = NgramModel(3, words, estimator)

# Apply the language model to generate 50 words in sequence
text_words = model.generate(50)

# Concatenate all words generated in a string separating them by a space.
text = ' '.join([word for word in text_words])

# print the text
print text

print model.prob('e', ['a', 't'])