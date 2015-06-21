"""
Inference
@author: Ke Zhai (zhaike@cs.umd.edu)
"""
import abc;
import numpy;
import scipy;
import scipy.special;

"""
"""
class Inference(object):
    __metaclass__ = abc.ABCMeta;

    """
    """
    def __init__(self,
                 snapshot_interval=10,
                 gamma_maximum_iteration=100,
                 alpha_maximum_iteration=100,
                 #converge_threshold = 0.00001,
                 #global_maximum_iteration=100
                 ):
        # initialize the iteration parameters
        self._alpha_maximum_iteration = alpha_maximum_iteration;
        self._gamma_maximum_iteration = gamma_maximum_iteration;
        
        #self._global_maximum_iteration = global_maximum_iteration;
        #self._converge_threshold = converge_threshold
        
        self._snapshot_iterval = snapshot_interval;
        
        self._gamma_title = "gamma-";
        self._beta_title = "beta-";
        
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, type_to_index, index_to_type, num_topics, alpha, beta):
        self._counter = 0;
        
        #self._type_to_index = type_to_index;
        self._index_to_type = index_to_type;
        
        # initialize the total number of topics.
        self._K = num_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        #self._alpha = numpy.random.random((1, self._K)) / self._K;
        self._alpha = numpy.zeros((1, self._K))+alpha;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        #from util.input_parser import dict_list_2_dict_freqdist
        #data = dict_list_2_dict_freqdist(data);
        self._data = data
        
        # initialize the size of the collection, i.e., total number of documents.
        self._number_of_documents = len(self._data)
        
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._vocab += token_list
        self._vocab = list(set(self._vocab))
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab)
        




    """
    """
    def print_topics(self, term_mapping, top_words=10):
        input = open(term_mapping);
        vocab = {};
        i = 0;
        for line in input:
            vocab[i] = line.strip();
            i += 1;

        if top_words >= self._V:
            sorted_beta = numpy.zeros((1, self._K)) - numpy.log(self._V);
        else:
            sorted_beta = numpy.sort(self._E_log_beta, axis=0);
            sorted_beta = sorted_beta[-top_words, :][numpy.newaxis, :];

        #print sorted_beta;
        
        #display = self._log_beta > -numpy.log(self._V);
        #assert(display.shape==(self._V, self._K));
        for k in xrange(self._K):
            display = self._E_log_beta[:, [k]] >= sorted_beta[:, k];
            assert(display.shape == (self._V, 1));
            output_str = str(k) + ": ";
            for v in xrange(self._V):
                if display[v, :]:
                    output_str += vocab[v] + "\t";
            print output_str
