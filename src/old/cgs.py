"""
@author: Jordan Boyd-Graber (jbg@umiacs.umd.edu)
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

from inference import Inference

import math, random;
import numpy
import scipy;
import util.log_math;

from collections import defaultdict
from nltk import FreqDist

"""
This is a python implementation of lda, based on collapsed Gibbs sampling, with hyper parameter updating.
It only supports symmetric Dirichlet prior over the topic simplex.

References:
[1] T. L. Griffiths & M. Steyvers. Finding Scientific Topics. Proceedings of the National Academy of Sciences, 101, 5228-5235, 2004.
"""
class CollapsedGibbsSampling(Inference):
    """
    """
    def __init__(self,
                 snapshot_interval=10,
                 gibbs_sampling_maximum_iteration=100,
                 alpha_maximum_iteration=100,
                 gamma_maximum_iteration=5,
                 hyper_parameter_sampling_interval=25):

        super(CollapsedGibbsSampling, self).__init__(snapshot_interval,
                                                     gamma_maximum_iteration,
                                                     alpha_maximum_iteration
                                                     );

        #self._alpha_maximum_iteration = alpha_maximum_iteration
        self._gibbs_sampling_maximum_iteration = gibbs_sampling_maximum_iteration
        self._hyper_parameter_sampling_interval = hyper_parameter_sampling_interval;
        assert(self._hyper_parameter_sampling_interval>0);
        
    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, data, type_to_index, index_to_type, num_topics=10, alpha=0.5, beta=0.1):
        super(CollapsedGibbsSampling, self)._initialize(data, type_to_index, index_to_type, num_topics, alpha, beta);

        # set the document smooth factor
        self._alpha = alpha
        # set the vocabulary smooth factor
        self._log_beta = beta
        
        # define the topic assignment for every word in every document, first indexed by doc id, then indexed by word position
        self._topic_assignment = defaultdict(dict)
        
        self._K = num_topics
    
        self._alpha_sum = self._alpha * self._K

        # define the input data
        self._data = data
        # define the total number of document
        self._number_of_documents = len(data)
    
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = set([])
        self._vocab = type_to_index;
        
        for doc in xrange(self._number_of_documents):
            #self._topic_assignment[doc] = numpy.random.randint(0, self._K, (1, len(self._data[doc])));
            self._topic_assignment[doc] = numpy.zeros((1, len(self._data[doc])))-1;

            #for position in xrange(len(self._data[doc])):
                # learn all the words we'll see
                #self._vocab.add(self._data[doc][position])
            
                # initialize the state to unassigned
                #self._topic_assignment[doc][position] = -1
                
        self._V = len(self._vocab)
        
        # define the counts over different topics for all documents, first indexed by doc id, the indexed by topic id
        #self._document_topic_counts = defaultdict(FreqDist)
        self._document_topic_counts = numpy.zeros((self._number_of_documents, self._K))
        # define the counts over words for all topics, first indexed by topic id, then indexed by token id
        #self._topic_term_counts = defaultdict(FreqDist)
        self._topic_term_counts = numpy.zeros((self._K, self._V))

    """
    
    """
    def optimize_hyperparameters(self, samples=5, step=3.0):
        rawParam = [math.log(self._alpha), math.log(self._log_beta)]

        for ii in xrange(samples):
            log_likelihood_old = self.compute_likelihood(self._alpha, self._log_beta)
            log_likelihood_new = math.log(random.random()) + log_likelihood_old
            #print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha, self._log_beta))

            l = [x - random.random() * step for x in rawParam]
            r = [x + step for x in rawParam]

            for jj in xrange(self._alpha_maximum_iteration):
                rawParamNew = [l[x] + random.random() * (r[x] - l[x]) for x in xrange(len(rawParam))]
                trial_alpha, trial_beta = [math.exp(x) for x in rawParamNew]
                lp_test = self.compute_likelihood(trial_alpha, trial_beta)

                if lp_test > log_likelihood_new:
                    #print(jj)
                    self._alpha = math.exp(rawParamNew[0])
                    self._log_beta = math.exp(rawParamNew[1])
                    self._alpha_sum = self._alpha * self._K
                    self._beta_sum = self._log_beta * self._V
                    rawParam = [math.log(self._alpha), math.log(self._log_beta)]
                    break
                else:
                    for dd in xrange(len(rawParamNew)):
                        if rawParamNew[dd] < rawParam[dd]:
                            l[dd] = rawParamNew[dd]
                        else:
                            r[dd] = rawParamNew[dd]
                        assert l[dd] <= rawParam[dd]
                        assert r[dd] >= rawParam[dd]

            #print("\nNew hyperparameters (%i): %f %f" % (jj, self._alpha, self._log_beta))

    """
    compute the log-likelihood of the model
    """
    def compute_likelihood(self, alpha, beta):
        #assert len(self._do_number_of_documentstopics) == self._number_of_documents
        
        alpha_sum = alpha * self._K
        beta_sum = beta * self._V

        likelihood = 0.0
        # compute the log likelihood of the document
        likelihood += scipy.special.gammaln(alpha_sum) * len(self._data)
        likelihood -= scipy.special.gammaln(alpha) * self._K * len(self._data)
        
        likelihood += numpy.sum(scipy.special.gammaln(self._document_topic_counts + alpha));
        likelihood -= numpy.sum(scipy.special.gammaln(alpha_sum + numpy.sum(self._document_topic_counts, axis=1)));
        #for ii in xrange(self._number_of_documents):#self._document_topic_counts.keys():
            #for jj in xrange(self._K):
                #likelihood += scipy.special.gammaln(alpha + self._document_topic_counts[ii][jj])
            #likelihood -= scipy.special.gammaln(alpha_sum + self._document_topic_counts[ii].N())
            
        # compute the log likelihood of the topic
        likelihood += scipy.special.gammaln(beta_sum) * self._K
        likelihood -= scipy.special.gammaln(beta) * self._V * self._K
            
        likelihood += numpy.sum(scipy.special.gammaln(beta + self._topic_term_counts));
        likelihood -= numpy.sum(scipy.special.gammaln(beta_sum + numpy.sum(self._topic_term_counts, axis=1)));
        #for ii in self._topic_term_counts.keys():
            #for jj in self._vocab:
                #likelihood += scipy.special.gammaln(beta + self._topic_term_counts[ii][jj])
            #likelihood -= scipy.special.gammaln(beta_sum + self._topic_term_counts[ii].N())
            
        return likelihood

    """
    compute the conditional distribution
    @param doc: doc id
    @param word: word id
    @param topic: topic id  
    @return: the probability value of the topic for that word in that document
    """
    def log_prob(self, doc, word, topic):
        val = math.log(self._document_topic_counts[doc][topic] + self._alpha)
        #this is constant across a document, so we don't need to compute this term
        # val -= math.log(self._document_topic_counts[doc].N() + self._alpha_sum)
        
        val += math.log(self._topic_term_counts[topic][word] + self._log_beta)
        val -= math.log(self._topic_term_counts[topic].N() + self._V * self._log_beta)
    
        return val

    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._topic_assignment, self._document_topic_counts and self._topic_term_counts will change
    @param doc: a document id
    @param position: the position in doc, ranged as range(self._data[doc])
    """
    def sample_word(self, doc):
        for position in xrange(len(self._data[doc])):
            assert position >= 0 and position < len(self._data[doc])
            
            #retrieve the word_id
            word_id = self._data[doc][position]
        
            #get the old topic assignment to the word_id in doc at position
            old_topic = self._topic_assignment[doc][0, position]
            if old_topic != -1:
                #this word_id already has a valid topic assignment, decrease the topic|doc counts and word_id|topic counts by covering up that word_id
                self.change_count(doc, word_id, old_topic, -1)
    
            #compute the topic probability of current word_id, given the topic assignment for other words
            probs = self._document_topic_counts[[doc], :] + self._alpha

            probs *= self._topic_term_counts[:, [word_id]].T + self._log_beta
            probs /= numpy.sum(self._topic_term_counts, axis=1)[:, numpy.newaxis].T + self._V * self._log_beta;
            probs /= numpy.sum(probs);
            #probs = [self.log_prob(doc, self._data[doc][position], x) for x in xrange(self._K)]
    
            #sample a new topic out of a distribution according to probs
            #new_topic = util.log_math.log_sample(probs)
            
            new_topic = numpy.nonzero(numpy.random.multinomial(1, probs[0, :])==1)[0][0];
    
            #after we draw a new topic for that word_id, we will change the topic|doc counts and word_id|topic counts, i.e., add the counts back
            self.change_count(doc, word_id, new_topic, 1)
            #assign the topic for the word_id of current document at current position
            self._topic_assignment[doc][0, position] = new_topic

    """
    this methods change the count of a topic in one doc and a word of one topic by delta
    this values will be used in the computation
    @param doc: the doc id
    @param word: the word id
    @param topic: the topic id
    @param delta: the change in the value
    """
    def change_count(self, doc, word, topic, delta):
        #self._document_topic_counts[doc].inc(topic, delta)
        #self._topic_term_counts[topic].inc(word, delta)
        self._document_topic_counts[doc, topic] += delta
        self._topic_term_counts[topic, word] += delta

    """
    sample the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def sample(self):
        assert self._topic_assignment
        
        #sample the total corpus
        for iter in xrange(self._gibbs_sampling_maximum_iteration):
            #sample every document
            for doc in xrange(self._number_of_documents):
                #sample every position
                for iter2 in xrange(self._gamma_maximum_iteration):
                    #for position in xrange(len(self._data[doc])):
                    self.sample_word(doc)
                    
            print("iteration %i %f" % (iter, self.compute_likelihood(self._alpha, self._log_beta)))
            if iter % self._hyper_parameter_sampling_interval == 0:
                self.optimize_hyperparameters()

    def print_topics(self, num_words=15):
        for ii in self._topic_term_counts:
            print("%i:%s\n" % (ii, "\t".join(self._index_to_type(item) for item in self._topic_term_counts[ii].keys()[:num_words])))

if __name__ == "__main__":
    temp_directory = "../data/ap/";
    from launch import parse_data;
    doc, type_to_index, index_to_type = parse_data(temp_directory+"doc.dat");
    
    lda = CollapsedGibbsSampling()
    lda._initialize(doc, type_to_index, index_to_type, 10)

    lda.sample()
    lda.print_topics()