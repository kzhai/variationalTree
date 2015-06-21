from nltk.corpus import europarl_raw
from pylda import *
from pyldawn import *
from itertools import chain
from glob import glob

def flatten(list_of_lists):
    return chain.from_iterable(list_of_lists)

def europarl_demo(num_topics = 10, doc_limit = 500, token_limit=250):
    doc = 0
    vocab_builder = VocabBuilder()

    for ii in europarl_raw.english.chapters():
        vocab_builder.scan(flatten(ii))
        doc += 1
        if doc_limit > 0 and doc > doc_limit:
            break

    vocab = vocab_builder.vocab()
    lda = Sampler(num_topics, vocab)

    print("Created vocab: %s... (%i words)" % (str(vocab[:40]), len(vocab)))

    for ii in europarl_raw.english.chapters():
        doc = lda.add_doc(flatten(ii), vocab, token_limit)
        if doc_limit > 0 and doc > doc_limit:
            break

    print("Added %i documents" % doc)

    lda.run_sampler(500)

    lda.report_topics(vocab)

def getVocab(vocab_filename):
    vocab = []
    vocab_file = open(vocab_filename, 'r')
    for line in vocab_file:
        line = line.strip()
        words = line.split('\t')
        vocab.append(words[1])
    return vocab

def readFile(file_name):
    print file_name
    doc_file = open(file_name, 'r')
    text = ''
    for line in doc_file:
        line = line.strip()
        text += line + ' '

    text = text.strip()
    words = text.split(' ')
    return words

def demo_20_news(num_topics = 20, doc_limit = -1, token_limit = 1000):

    vocab_filename = '../ldawn/vocab/20_news_stem_tfidf.voc'
    tree_files = '../ldawn/wn/20_news_stem_tfidf_empty.wn.*'
    hyper_file = '../ldawn/hyperparameters/tree_hyperparams'
    data_file = '../../data/20_news_date/preprocessed/*'

    # read vocab
    vocab = getVocab(vocab_filename)

    # initialize sampler
    ldawn = ldawnSampler(num_topics, vocab, tree_files, hyper_file)
	
    print len(glob(data_file))
    # read in documents
    count = -1
    for file_name in glob(data_file):
        words = readFile(file_name)
        count += 1
        doc = ldawn.add_doc(words, vocab, doc_id=count)
        if doc_limit > 0 and (count+1) >= doc_limit:
            break

    print("Added %i documents" % (count+1))

    ldawn.run_sampler(100)
    ldawn.report_topics(vocab, 15)


def demo_toy_ldawn(num_topics = 3, doc_limit = 10, token_limit = 10):

    vocab_filename = '../ldawn/vocab/toy.voc'
    tree_files = '../ldawn/wn/toy_empty.wn.*'
    #tree_files = '../ldawn/wn/toy.wn.*'
    hyper_file = '../ldawn/hyperparameters/tree_hyperparams'
    data_file = '../../data/toy/*'

    # read vocab
    vocab = getVocab(vocab_filename)

    # initialize sampler
    ldawn = ldawnSampler(num_topics, vocab, tree_files, hyper_file)
	
    print len(glob(data_file))
    # read in documents
    count = -1
    for file_name in glob(data_file):
        words = readFile(file_name)
        count += 1
        print count
        doc = ldawn.add_doc(words, vocab, doc_id=count)
        if doc_limit > 0 and doc > doc_limit:
            break

    print("Added %i documents" % (doc+1))

    ldawn.run_sampler(100)
    ldawn.report_topics(vocab, 10)

'''
def demo_toy_lda(num_topics = 3, doc_limit = 10, token_limit = 10):

    vocab_filename = '../ldawn/vocab/toy.voc'
    data_file = '../../data/toy/*'

    # read vocab
    vocab = getVocab(vocab_filename)

    # initialize sampler
    lda = Sampler(num_topics, vocab)
	
    print len(glob(data_file))
    # read in documents
    count = -1
    for file_name in glob(data_file):
        words = readFile(file_name)
        count += 1
        print count
        doc = lda.add_doc(words, vocab, doc_id=count, rand_stub=count+1)
        if doc_limit > 0 and doc > doc_limit:
            break

    print("Added %i documents" % (doc+1))

    lda.run_sampler(10)
    lda.report_topics(vocab, 10)
'''

if __name__ == "__main__":
    #europarl_demo(doc_limit=50)
    #demo_20_news()
    #demo_toy_lda()
    demo_toy_ldawn()
