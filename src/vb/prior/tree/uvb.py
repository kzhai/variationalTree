"""
UncollapsedVariationalBayes for Vanilla LDA with Tree Prior
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy
import nltk
import sys
import codecs;

try:
    import cPickle as pickle
except:
    import pickle

#from util.log_math import log_normalize
from collections import defaultdict;

"""
This is a python implementation of vanilla lda with tree prior, based on variational inference, with hyper parameter updating.

References:
[1] Y. Hu, J. Boyd-Graber, and B. Satinoff. Interactive Topic Modeling. Association for Computational Linguistics (ACL), 2011.
"""

def parse_data(documents_file, vocabulary_file=None):
    import codecs
    
    type_to_index = {};
    index_to_type = {};
    vocabulary = [];
    if (vocabulary_file!=None):
        input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
        for line in input_file:
            #line = line.strip().split()[0];
            assert len(line.strip().split())==4;
            line = line.strip().split()[1];
            assert line not in type_to_index, "duplicate type for %s" % line;
            type_to_index[line] = len(type_to_index);
            index_to_type[len(index_to_type)] = line;
            vocabulary.append(line);
        input_file.close();

    input_file = codecs.open(documents_file, mode="r", encoding="utf-8")

    doc_count = 0
    documents = []
    
    for line in input_file:
        line = line.strip().lower();

        contents = line.split("\t");

        document = nltk.probability.FreqDist();
        for token in contents[-1].split():
            if token not in type_to_index:
                if vocabulary_file==None:
                    type_to_index[token] = len(type_to_index);
                    index_to_type[len(index_to_type)] = token;
                else:
                    continue;
                    
            document.inc(type_to_index[token]);
            #document.append(type_to_index[token]);
        
        assert len(document)>0, "document %d collapsed..." % doc_count;
        
        documents.append(document);
        
        #if len(contents)==2:
            #documents[int(contents[0])] = document;
        #elif len(contents)==1:
            #documents[doc_count] = document;
        #else:
            #print ""

        doc_count+=1
        if doc_count%10000==0:
            print "successfully import %d documents..." % doc_count;
    
    input_file.close();

    print "successfully import", len(documents), "documents..."
    return documents, type_to_index, index_to_type, vocabulary

from inferencer import Inferencer;
from inferencer import compute_dirichlet_expectation;
class UncollapsedVariationalBayes(Inferencer):
    """
    """
    def __init__(self,
                 update_hyper_parameter=True,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001,
                 gamma_converge_threshold=0.000001,
                 gamma_maximum_iteration=20
                 ):
        Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        #self._alpha_update_decay_factor = alpha_update_decay_factor;
        #self._alpha_maximum_decay = alpha_maximum_decay;
        #self._alpha_converge_threshold = alpha_converge_threshold;
        #self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        self._gamma_maximum_iteration = gamma_maximum_iteration;
        self._gamma_converge_threshold = gamma_converge_threshold;

        #self._model_likelihood_threshold = model_likelihood_threshold;

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    '''
    def _initialize(self, data, prior_tree, type_to_index, index_to_type, number_of_topics, alpha):
        
        self._counter = 0;
        
        self._type_to_index = type_to_index;
        self._index_to_type = index_to_type;
        
        # initialize the total number of topics.
        self._number_of_topics = number_of_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        #self._alpha = numpy.random.random((1, self._number_of_topics)) / self._number_of_topics;
        self._alpha = numpy.zeros((1, self._number_of_topics))+alpha;
        #self._eta = eta;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._data = data

        # initialize the size of the collection, i.e., total number of documents.
        self._number_of_documents = len(self._data)
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        #self._number_of_terms = len(self._type_to_index)
        
        self.update_tree_structure(prior_tree);
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        #self._gamma = numpy.zeros((self._number_of_documents, self._number_of_topics)) + self._alpha + 1.0 * self._number_of_paths / self._number_of_topics;
        #self._gamma = numpy.tile(self._alpha + 1.0 * self._number_of_terms / self._number_of_topics, (self._number_of_documents, 1));
        self._gamma = self._alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        
        # initialize a _E_log_beta variable, indexed by node, valued by a K-by-C matrix, where C stands for the number of children of that node
        self._E_log_beta = numpy.random.gamma(100., 1./100., (self._number_of_topics, self._number_of_edges));
        for node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[node_index];
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
    '''

    def e_step(self):
        document_level_log_likelihood = 0;

        # initialize a V-by-K matrix phi contribution
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_paths));
        
        # iterate over all documents
        for doc_id in xrange(self._number_of_documents):
        #for doc_id in xrange(0, 1):
            # compute the total number of words
            #total_word_count = self._data[doc_id].N()

            # initialize gamma for this document
            #self._gamma[[doc_id], :] = self._alpha + 1.0 * total_word_count / self._number_of_topics;
            #self._gamma[[doc_id], :] = self._alpha + 2.0 * total_word_count / self._number_of_topics * numpy.random.random((1, self._number_of_topics));
            
            #word_ids = numpy.array(self._data[doc_id].keys());
            #word_counts = numpy.array([self._data[doc_id].values()]);
            #assert(word_counts.shape == (1, len(word_ids)));

            # update phi and gamma until gamma converges
            for gamma_iteration in xrange(self._gamma_maximum_iteration):
                document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                
                phi_entropy = 0;
                phi_E_log_beta = 0;

                #E_log_theta = scipy.special.psi(self._gamma[[doc_id], :]).T;
                #assert E_log_theta.shape==(self._number_of_topics, 1);
                
                E_log_theta = compute_dirichlet_expectation(self._gamma[[doc_id], :]).T;
                assert E_log_theta.shape==(self._number_of_topics, 1);

                for word_id in self._data[doc_id].keys():#word_ids:
                    paths_lead_to_current_word = self._word_index_to_path_indices[word_id];

                    #log_phi = numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]).T, (1, len(paths_lead_to_current_word)));
                    log_phi = numpy.tile(E_log_theta, (1, len(paths_lead_to_current_word)));
                    assert log_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));

                    for position_index in xrange(len(paths_lead_to_current_word)):
                        log_phi[:, position_index] += numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[position_index]]], axis=1);
                    del position_index

                    # log normalize
                    # TODO: error with 2-level tree
                    #log_phi = log_normalize(log_phi)
                    
                    # convert it into normal scale
                    path_phi = numpy.exp(log_phi - numpy.max(log_phi));
                    assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                    assert numpy.min(path_phi)>=0; path_phi.T
                    # normalize path_phi over all topics and paths
                    assert numpy.sum(path_phi)>0, log_phi.T
                    
                    path_phi /= numpy.sum(path_phi);
                    #path_phi /= numpy.sum(path_phi, axis=1);
                    
                    phi_entropy += - numpy.sum(path_phi * numpy.log(path_phi)) * self._data[doc_id][word_id];
                    
                    for position_index in xrange(len(paths_lead_to_current_word)):
                        phi_E_log_beta += self._data[doc_id][word_id] * numpy.sum( path_phi[:, [position_index]] * numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[position_index]]], axis=1)[:, numpy.newaxis] )
                    del position_index
                    
                    # multiple path_phi with the count of current word
                    path_phi *= self._data[doc_id][word_id];
                    
                    document_phi[:, paths_lead_to_current_word] += path_phi;
                
                del word_id, paths_lead_to_current_word
                #print doc_id, "before", self._gamma[[doc_id], :];
                self._gamma[[doc_id], :] = self._alpha + numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T;
                
            # term 1
            document_level_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha)) - numpy.sum(scipy.special.gammaln(self._alpha));
            document_level_log_likelihood += numpy.sum((self._alpha - 1) * compute_dirichlet_expectation(self._gamma[[doc_id], :]));

            # term 2
            document_level_log_likelihood += numpy.sum(numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T * compute_dirichlet_expectation(self._gamma[[doc_id], :]));
            
            # term 4
            document_level_log_likelihood += phi_E_log_beta;
            
            # term 5
            document_level_log_likelihood += - scipy.special.gammaln(numpy.sum(self._gamma[[doc_id], :])) + numpy.sum(scipy.special.gammaln(self._gamma[[doc_id], :]))
            document_level_log_likelihood += - numpy.sum( (self._gamma[[doc_id], :] - 1) * compute_dirichlet_expectation(self._gamma[[doc_id], :]) );
            
            # term 7
            document_level_log_likelihood += phi_entropy;
            
            phi_sufficient_statistics += document_phi;
        
            if (doc_id+1) % 1000==0:
                print "successfully processed %d documents..." % (doc_id+1);
                
            del doc_id
        
        return phi_sufficient_statistics, document_level_log_likelihood;

    '''
    def m_step(self, phi_sufficient_statistics):
        assert phi_sufficient_statistics.shape==(self._number_of_topics, self._number_of_paths);
        assert numpy.min(phi_sufficient_statistics)>=0, phi_sufficient_statistics;
        
        var_beta = numpy.tile(self._edge_prior, (self._number_of_topics, 1))
        assert var_beta.shape==(self._number_of_topics, self._number_of_edges);
        assert numpy.min(var_beta)>=0;
        #for internal_node_index in self._edges_from_internal_node:
            #edges_indices_list = self._edges_from_internal_node[internal_node_index];
            #var_beta[:, edges_indices_list] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_internal_node[internal_node_index]], axis=1)[:, numpy.newaxis];
        
        for edge_index in self._index_to_edge:
            #print var_beta[:, edge_index].shape, numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis].shape;
            var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
        del edge_index
        assert(var_beta.shape == (self._number_of_topics, self._number_of_edges));
        assert numpy.min(var_beta)>=0;

        self._E_log_beta = numpy.copy(var_beta);
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            assert numpy.min(self._E_log_beta[:, edge_index_list])>=0;
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
        del internal_node_index, edge_index_list;
        assert numpy.min(var_beta)>=0;

        corpus_level_log_likelihood = 0;
        for internal_node_index in self._edges_from_internal_node:
            edges_indices_list = self._edges_from_internal_node[internal_node_index];
            
            # term 3
            tmp = corpus_level_log_likelihood
            corpus_level_log_likelihood += (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list]))) * self._number_of_topics;
            corpus_level_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T));
            
            #print numpy.sum(var_beta[:, edges_indices_list].T)
            #print numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T)
            
            # term 6
            #corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)))
            #corpus_level_log_likelihood += numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]));
            corpus_level_log_likelihood += numpy.sum( - scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)) + numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]), axis=1));
            corpus_level_log_likelihood += numpy.sum( - (var_beta[:, edges_indices_list]-1) * compute_dirichlet_expectation(var_beta[:, edges_indices_list]));
                    
        assert numpy.min(var_beta)>=0;

        # TODO: add in alpha updating
        # compute the sufficient statistics for alpha and update
        #alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        #alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
        #self.update_alpha(alpha_sufficient_statistics)
        
        #print numpy.sum(numpy.exp(self._E_log_beta), axis=1);
        
        return corpus_level_log_likelihood
    '''
    
    """
    """
    
    '''
    def learn(self):
        self._counter += 1;
        
        clock_e_step = time.time();        
        phi_sufficient_statistics, document_level_log_likelihood = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();        
        corpus_level_log_likelihood = self.m_step(phi_sufficient_statistics);
        clock_m_step = time.time() - clock_m_step;
                
        # compute the log-likelihood of alpha terms
        #alpha_sum = numpy.sum(self._alpha, axis=1);
        #likelihood_alpha = -numpy.sum(scipy.special.gammaln(self._alpha), axis=1);
        #likelihood_alpha += scipy.special.gammaln(alpha_sum);
        #likelihood_alpha *= self._number_of_documents;
        
        #likelihood_gamma = numpy.sum(scipy.special.gammaln(self._gamma));
        #likelihood_gamma -= numpy.sum(scipy.special.gammaln(numpy.sum(self._gamma, axis=1)));

        #new_likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi;
        
        new_likelihood = document_level_log_likelihood + corpus_level_log_likelihood;
        
        print "e_step and m_step of iteration %d finished in %d and %d seconds respectively with log likelihood %g" % (self._counter, clock_e_step, clock_m_step, new_likelihood)
        print "document likelihood: ", document_level_log_likelihood
        print "corpus likelihood: ", corpus_level_log_likelihood
        #if abs((new_likelihood - old_likelihood) / old_likelihood) < self._model_likelihood_threshold:
            #print "model likelihood converged..."
            #break
        #old_likelihood = new_likelihood;
        
        return new_likelihood
    '''
    
    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """
    '''
    def update_alpha(self, alpha_sufficient_statistics):
        assert(alpha_sufficient_statistics.shape == (1, self._number_of_topics));        
        alpha_update = self._alpha;
        
        decay = 0;
        for alpha_iteration in xrange(self._alpha_maximum_iteration):
            alpha_sum = numpy.sum(self._alpha);
            alpha_gradient = self._number_of_documents * (scipy.special.psi(alpha_sum) - scipy.special.psi(self._alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -self._number_of_documents * scipy.special.polygamma(1, self._alpha);

            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self._number_of_documents * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(self._alpha_update_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                #print "step size is", step_size
                assert(self._alpha.shape == step_size.shape);
                
                if numpy.any(self._alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > self._alpha_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha));
            self._alpha = alpha_update;
            if mean_change <= self._alpha_converge_threshold:
                break;

        return
    '''
    
    '''
    def export_topic_term_distribution(self, exp_beta_path):
        output = open(exp_beta_path, 'w');
        
        #term_path_ranking = defaultdict(nltk.probability.FreqDist);
        for k in xrange(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (k));

            freqdist = nltk.probability.FreqDist()
            for path_index in self._path_index_to_word_index:
                path_rank = 1;
                for edge_index in self._edges_along_path[path_index]:
                    path_rank *= numpy.exp(self._E_log_beta[k, edge_index]);
                    
                freqdist.inc(path_index, path_rank);
            
            for key in freqdist.keys():
                output.write("%s\t%g\n" % (self._index_to_type[self._path_index_to_word_index[key]], freqdist[key]));
                
        output.close();

    def dump_tree(self, output_tree_file):
        output_stream = codecs.open(output_tree_file, 'w', 'utf-8');

        pickle.dump(self._maximum_depth, output_stream);

        pickle.dump(self._edge_to_index, output_stream);
        pickle.dump(self._index_to_edge, output_stream);

        #pickle.dump(self._number_of_edges, output_stream);

        pickle.dump(self._edges_from_internal_node, output_stream);

        pickle.dump(self._edge_prior, output_stream);
        
        output_stream.flush();

        pickle.dump(self._path_index_to_word_index, output_stream);
        pickle.dump(self._word_index_to_path_indices, output_stream);

        #pickle.dump(self._number_of_paths, output_stream);

        pickle.dump(self._edges_along_path, output_stream);
        pickle.dump(self._paths_through_edge, output_stream);
        
        output_stream.flush();
        output_stream.close();

    def dump_parameters(self, output_parameter_file):
        # this pops up error
        output_stream = codecs.open(output_parameter_file, 'w', 'utf-8');
        
        pickle.dump(self._counter, output_stream);
        
        pickle.dump(self._type_to_index, output_stream);
        pickle.dump(self._index_to_type, output_stream);

        #print >> sys.stderr, "dump", self._index_to_type[0], self._index_to_type[1], self._index_to_type[2], self._index_to_type[3]

        pickle.dump(self._number_of_topics, output_stream);
        pickle.dump(self._number_of_edges, output_stream);
        pickle.dump(self._number_of_paths, output_stream);
        
        pickle.dump(self._alpha, output_stream);
        
        #pickle.dump(self._hybrid_mode, output_stream);
        
        output_stream.flush();
        output_stream.close();
        
    def load_params(self, input_parameter_file):
        input_stream = codecs.open(input_parameter_file, 'r');

        self._counter = pickle.load(input_stream);
        
        self._type_to_index = pickle.load(input_stream);
        self._index_to_type = pickle.load(input_stream);

        #tmp = unicode(ww.term_str).encode('utf-8')
        #print >> sys.stderr, len(self._type_to_index), self._type_to_index
        #print >> sys.stderr, "load", self._index_to_type[0];
        #print >> sys.stderr, "load", unicode(self._index_to_type[0]).encode('utf-8')

        self._number_of_topics = pickle.load(input_stream);
        self._number_of_edges = pickle.load(input_stream);
        self._number_of_paths = pickle.load(input_stream);
        
        self._alpha = pickle.load(input_stream);
        
        #self._hybrid_mode = pickle.load(input_stream);
        
        input_stream.close();
        
        print >> sys.stderr, "successfully load params from %s..." % input_parameter_file
        
    def load_tree(self, input_tree_file):
        input_stream = codecs.open(input_tree_file, 'r');

        self._maximum_depth = pickle.load(input_stream);

        self._edge_to_index = pickle.load(input_stream);
        self._index_to_edge = pickle.load(input_stream);
        
        #self._number_of_edges = pickle.load(input_stream);

        self._edges_from_internal_node = pickle.load(input_stream);

        self._edge_prior = pickle.load(input_stream);
        
        self._path_index_to_word_index = pickle.load(input_stream);
        self._word_index_to_path_indices = pickle.load(input_stream);
        
        #self._number_of_paths = pickle.load(input_stream);

        self._edges_along_path = pickle.load(input_stream);
        self._paths_through_edge = pickle.load(input_stream);
        
        input_stream.close();
        
        print >> sys.stderr, "successfully load tree from %s..." % input_tree_file

    def dump_E_log_beta(self, output_beta_file):
        numpy.savetxt(output_beta_file, self._E_log_beta);
        
    def load_E_log_beta(self, input_beta_file):
        self._E_log_beta = numpy.loadtxt(input_beta_file);
        print >> sys.stderr, "successfully load E-log-beta from %s..." % input_beta_file
        
    def dump_gamma(self, output_gamma_file):
        output_stream = codecs.open(output_gamma_file, 'w');
        pickle.dump(self._gamma, output_stream);
        output_stream.flush();
        output_stream.close();
        
        #numpy.savetxt(output_gamma_file, self._gamma);
        
    def load_gamma(self, input_gamma_file):
        input_stream = codecs.open(input_gamma_file, 'r');
        self._gamma = pickle.load(input_stream);
        input_stream.close();
        
        print >> sys.stderr, "successfully load gamma from %s..." % input_gamma_file

        #self._gamma = numpy.loadtxt(input_gamma_file);
    '''

if __name__ == "__main__":
    raise NotImplementedError;
