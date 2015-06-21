# Import the corpus and functions used from nltk library  
from nltk.corpus import brown;
from nltk.corpus import genesis
from nltk.probability import LidstoneProbDist
from nltk.model import NgramModel
  
# Tokens contains the words for Genesis and Reuters Trade  
#tokens = list(genesis.words('english-kjv.txt'))

#tokens.extend(list(reuters.words(categories = 'trade')))

# estimator for smoothing the N-gram model
estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)

# N-gram language model with 3-grams
#model = NgramModel(3, tokens, estimator)
model = NgramModel(3, brown.words(categories='news'), estimator)
#model = NgramModel(3, tokens)

# Apply the language model to generate 50 words in sequence
text_words = model.generate(50)

# Concatenate all words generated in a string separating them by a space.
text = ' '.join([word for word in text_words])

# print the text
print text

print model.prob('repayments', ['international', 'debt']);