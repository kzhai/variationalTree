"""
UncollapsedVariationalBayes for Vanilla LDA with Tree Prior
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import time
import numpy
import scipy
import nltk;
import sys;
import codecs

from collections import defaultdict;

"""
This is a python implementation of vanilla lda with tree prior, based on variational inference, with hyper parameter updating.

References:
[1] Y. Hu, J. Boyd-Graber, and B. Satinoff. Interactive Topic Modeling. Association for Computational Linguistics (ACL), 2011.
"""

def parse_data(documents_file, vocabulary_file=None):
    '''
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
    '''

    #'''
    type_to_index = {};
    index_to_type = {};
    #vocabulary = [];
    if (vocabulary_file!=None):
        input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
        for line in input_file:
            #line = line.strip().split()[0];
            assert len(line.strip().split())==4;
            line = line.strip().split()[1];
            assert line not in type_to_index, "duplicate type for %s" % line;
            type_to_index[line] = len(type_to_index);
            index_to_type[len(index_to_type)] = line;
            #vocabulary.append(line);
        input_file.close();

    #type_to_index = {};
    #index_to_type = {};
    vocabulary = [];
    if (vocabulary_file!=None):
        input_file = open(vocabulary_file, mode='r');
        for line in input_file:
            #line = line.strip().split()[0];
            assert len(line.strip().split())==4;
            line = line.strip().split()[1];
            #assert line not in type_to_index, "duplicate type for %s" % line;
            #type_to_index[line] = len(type_to_index);
            #index_to_type[len(index_to_type)] = line;
            vocabulary.append(line);
        input_file.close();
    #'''

    input_file = codecs.open(documents_file, mode="r", encoding="utf-8")
    doc_count = 0
    documents = []
    
    for line in input_file:
        line = line.strip().lower();

        contents = line.split("\t");

        document = [];
        for token in contents[-1].split():
            if token not in type_to_index:
                if vocabulary_file==None:
                    type_to_index[token] = len(type_to_index);
                    index_to_type[len(index_to_type)] = token;
                else:
                    continue;
                
            document.append(type_to_index[token]);
            #document.inc(type_to_index[token]);
            #document.append(type_to_index[token]);
        
        #assert len(document)>0, "document %d collapsed..." % doc_count;

        documents.append(document);
        
        doc_count+=1
        if doc_count%10000==0:
            print "successfully import %d documents..." % doc_count;
    
    input_file.close();

    print "successfully import", len(documents), "documents..."
    return documents, type_to_index, index_to_type, vocabulary

from inferencer import Inferencer;
from inferencer import compute_dirichlet_expectation;
class Hybrid(Inferencer):
    """
    """
    def __init__(self,
                 update_hyper_parameter=True,
                 alpha_update_decay_factor=0.9,
                 alpha_maximum_decay=10,
                 alpha_converge_threshold=0.000001,
                 alpha_maximum_iteration=100,
                 model_likelihood_threshold=0.00001,
                 number_of_samples=10,
                 burn_in_samples=5
                 ):
        Inferencer.__init__(self, update_hyper_parameter, alpha_update_decay_factor, alpha_maximum_decay, alpha_converge_threshold, alpha_maximum_iteration, model_likelihood_threshold);
        
        #self._alpha_update_decay_factor = alpha_update_decay_factor;
        #self._alpha_maximum_decay = alpha_maximum_decay;
        #self._alpha_converge_threshold = alpha_converge_threshold;
        #self._alpha_maximum_iteration = alpha_maximum_iteration;
        
        self._number_of_samples = number_of_samples;
        self._burn_in_samples = burn_in_samples;
        
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
        #self._gamma = self._alpha + 2.0 * self._number_of_paths / self._number_of_topics * numpy.random.random((self._number_of_documents, self._number_of_topics));
        
        # initialize a _E_log_beta variable, indexed by node, valued by a K-by-C matrix, where C stands for the number of children of that node
        self._E_log_beta = numpy.random.gamma(100., 1./100., (self._number_of_topics, self._number_of_edges));
        for node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[node_index];
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
    '''

    """
    def update_tree_structure(self, prior_tree):
        self._maximum_depth = prior_tree._max_depth
        
        self._edge_to_index = {}
        self._index_to_edge = {}

        self._edges_from_internal_node = defaultdict(list);
        
        self._edge_prior = [];

        for parent_node in prior_tree._nodes.keys():
            node = prior_tree._nodes[parent_node]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                for position_index in xrange(len(node._children_offsets)):
                    child_node = node._children_offsets[position_index];
                    self._edge_to_index[(parent_node, child_node)] = len(self._edge_to_index);
                    self._index_to_edge[len(self._index_to_edge)] = (parent_node, child_node);
                    
                    self._edges_from_internal_node[parent_node].append(self._edge_to_index[(parent_node, child_node)]);
                    
                    self._edge_prior.append(node._transition_prior[position_index]);

        self._number_of_edges = len(self._edge_prior);
        
        assert(len(self._edge_to_index)==self._number_of_edges);
        assert(len(self._index_to_edge)==self._number_of_edges);
                    
        self._edge_prior = numpy.array(self._edge_prior)[numpy.newaxis, :];
        assert(self._edge_prior.shape==(1, self._number_of_edges));
        
        '''
        print "edge index mapping"
        for (parent_node, child_node) in self._edge_to_index:
            print parent_node, child_node, self._edge_to_index[(parent_node, child_node)];
        print "index edge mapping"
        for edge_index in self._index_to_edge:
            print edge_index, self._index_to_edge[edge_index];
        print "edge prior", self._edge_prior
        print "edges from internal node"
        for internal_node in self._edges_from_internal_node:
            print internal_node, self._edges_from_internal_node[internal_node]
        '''

        self._edges_along_path = defaultdict(list);
        self._paths_through_edge = defaultdict(list);

        # word list indexed by path node_index 
        self._path_index_to_word_index = [];
        self._word_index_to_path_indices = defaultdict(list);
        
        # set of path indices that passes through a node
        #self._paths_through_internal_node = defaultdict(list)

        # compute the prior for every path, initialize self._path_prior_sum_of_word, self._path_prior_of_word,        
        path_index = 0;
        # iterate over all leaf nodes in the prior_tree
        for word in prior_tree._word_paths.keys():
            
            # iterate over all paths in the path of the current word (leaf node)
            for path in xrange(len(prior_tree._word_paths[word])):
                self._path_index_to_word_index.append(word)
                self._word_index_to_path_indices[word].append(path_index);

                nodes_along_word_path = prior_tree._word_paths[word][path]._nodes
                
                # if current word (leaf node) contains multiple words, add a leaf node_index for each word
                #if nodes_along_word_path[-1] in word_leaf[word].keys():
                    #leaf_index = word_leaf[word][nodes_along_word_path[-1]]
                    #nodes_along_word_path.append(leaf_index)
                
                for position_index in xrange(len(nodes_along_word_path)-1):
                    parent_node = nodes_along_word_path[position_index];
                    child_node = nodes_along_word_path[position_index+1];

                    #if parent_node not in self._paths_through_internal_node.keys():
                        #self._paths_through_internal_node[parent_node] = set()
                    #self._paths_through_internal_node[parent_node].add(path_index)
                    #self._paths_through_internal_node[parent_node].append(path_index)
                    
                    self._edges_along_path[path_index].append(self._edge_to_index[(parent_node, child_node)]);
                    self._paths_through_edge[self._edge_to_index[(parent_node, child_node)]].append(path_index);
                    
                    #self._paths_through_edge[self._edge_to_index[(parent_node, child_node)]].append(path_index);

                '''                    
                for edge in zip(nodes_along_word_path[:-1], nodes_along_word_path[1:]):
                    edge_index = self._edge_to_index[edge];
                    
                    self._edges_along_path[path_index].append(edge_index);
                    self._paths_through_edge[edge_index].append(path_index);
                '''

                path_index += 1;
                    
                #self._word_paths[word][path_index] = nodes_along_word_path

        self._number_of_paths = len(self._path_index_to_word_index);
        
        '''
        print "path word mapping"
        for path_index in self._path_index_to_word_index:
            print path_index, self._path_index_to_word_index[path_index];
        print "word path mapping"
        for word_index in self._word_index_to_path_indices:
            print word_index, self._word_index_to_path_indices[word_index];
        print "edges along path"
        for path_index in self._edges_along_path:
            print path_index, self._edges_along_path[path_index];
        #print "paths through internal node"
        #for internal_node in self._paths_through_internal_node:
            #print internal_node, self._paths_through_internal_node[internal_node];
        '''


        '''
        self._edge_prior_at_node = defaultdict(dict)
        self._edge_prior_sum_at_node = {}

        self._path_prior_of_word = defaultdict(dict)
        self._path_prior_sum_of_word = {}

        # this data structure is to handle the case that one node contains multiple words
        leaf_index = len(prior_tree._nodes.keys()) - 1
        word_leaf = defaultdict(dict)

        # compute the scalar for every edge, initialize self._edge_prior_at_node and self._edge_prior_sum_at_node
        for node_index in prior_tree._nodes.keys():
            node = prior_tree._nodes[node_index]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                self._edge_prior_sum_at_node[node_index] = node._transition_scalar
                
                #self._number_of_edges += len(node._children_offsets);
                
                for child_index in xrange(len(node._children_offsets)):
                    child_index = node._children_offsets[child_index]
                    self._edge_prior_at_node[node_index][child_index] = node._transition_prior[child_index]

            # if the node is a leaf node and it contains multiple words.
            # if yes, set the prior according to the words count in this node
            # It is equal to changing a node containing multiple words to a node
            # containing multiple leaf node and each node contains only one word
            if len(node._words) > 1:
                assert len(node._children_offsets) == 0
                assert len(node._words) > 0
                self._edge_prior_sum_at_node[node_index] = node._transition_scalar
                
                # TODO: increase the total number of edge
                for child_index in range(0, len(node._words)):
                    word_index = node._words[child_index]
                    leaf_index += 1
                    word_leaf[word_index][node_index] = leaf_index
                    self._edge_prior_at_node[node_index][leaf_index] = node._transition_prior[child_index]

        # nodes list indexed by word node_index and path node_index
        self._word_paths = defaultdict(dict)
        
        # word list indexed by path node_index 
        self._path_index_to_word_index = [];
        self._word_index_to_path_indices = defaultdict(set);
        
        # set of path indices that passes through a node 
        self._paths_through_internal_node = defaultdict()

        # compute the prior for every path, initialize self._path_prior_sum_of_word, self._path_prior_of_word,        
        path_index = -1
        # iterate over all leaf nodes in the prior_tree
        for word in prior_tree._word_paths.keys():
            self._path_prior_sum_of_word[word] = 0
            
            # iterate over all paths in the path of the current word (leaf node)
            for path in xrange(len(prior_tree._word_paths[word])):
                path_index += 1

                self._path_index_to_word_index.append(word)
                self._word_index_to_path_indices[word].add(path_index);

                nodes_along_word_path = prior_tree._word_paths[word][path]._nodes
                
                # if current word (leaf node) contains multiple words, add a leaf node_index for each word
                if nodes_along_word_path[-1] in word_leaf[word].keys():
                    leaf_index = word_leaf[word][nodes_along_word_path[-1]]
                    nodes_along_word_path.append(leaf_index)

                prob = 1.0
                for node_index in xrange(len(nodes_along_word_path) - 1):
                    parent = nodes_along_word_path[node_index]
                    child = nodes_along_word_path[node_index + 1]
                    prob *= self._edge_prior_at_node[parent][child]
                self._path_prior_of_word[word][path_index] = prob
                self._path_prior_sum_of_word[word] += prob
                    
                for node_index in nodes_along_word_path:
                    if node_index not in self._paths_through_internal_node.keys():
                        self._paths_through_internal_node[node_index] = set()
                    self._paths_through_internal_node[node_index].add(path_index)

                self._word_paths[word][path_index] = nodes_along_word_path

        self._number_of_paths = len(self._path_index_to_word_index);
        '''

        '''
        self._E_log_beta = defaultdict();

        for node_index in prior_tree._nodes.keys():
            node = prior_tree._nodes[node_index]
            
            # if the node is an internal node, compute the prior scalar for every edge
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                
                self._E_log_beta[node_index] = compute_dirichlet_expectation(numpy.random.gamma(100., 1./100., (self._number_of_topics, len(node._children_offsets))));
        '''

        #self._E_log_beta = compute_dirichlet_expectation(numpy.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_terms)));
    """

    def e_step(self):
        document_level_log_likelihood = 0;

        # initialize a dictionary store the topic distribution per document
        #gamma = {};        
        
        # initialize a V-by-K matrix path_phi contribution
        phi_sufficient_statistics = numpy.zeros((self._number_of_topics, self._number_of_paths));
        #alpha_sufficient_statistics = numpy.zeros((1, self._number_of_topics));
        
        # iterate over all documents
        for doc_id in xrange(self._number_of_documents):
        #for doc_id in xrange(10):
            #print self._data[doc_id]
            #continue;

            # compute the total number of words
            #total_word_count = self._data[doc_id].N()

            # initialize phi_sum for this document
            #self._gamma[[doc_id], :] = self._alpha + 1.0 * total_word_count / self._number_of_topics;
            #self._gamma[[doc_id], :] = self._alpha + 2.0 * total_word_count / self._number_of_topics * numpy.random.random((1, self._number_of_topics));

            topic_path_assignment = {};
            topic_sum = numpy.zeros((1, self._number_of_topics));
            for word_index in xrange(len(self._data[doc_id])):
                topic_assignment = numpy.random.randint(0, self._number_of_topics);
                path_assignment = numpy.random.randint(0, len(self._word_index_to_path_indices[self._data[doc_id][word_index]]));
                topic_path_assignment[word_index] = (topic_assignment, path_assignment);
                topic_sum[0, topic_assignment] += 1;
            #del word_index, topic_assignment, path_assignment;

            # update path_phi and phi_sum until phi_sum converges
            for sample_index in xrange(self._number_of_samples):
                #document_phi = numpy.zeros((self._number_of_topics, self._number_of_paths));
                
                phi_entropy = 0;
                phi_E_log_beta = 0;
                
                for word_index in xrange(len(self._data[doc_id])):
                    word_id = self._data[doc_id][word_index];
                    topic_sum[0, topic_path_assignment[word_index][0]] -= 1;
                    
                    paths_lead_to_current_word = self._word_index_to_path_indices[word_id];
                    assert len(paths_lead_to_current_word)>0

                    #path_phi = numpy.tile(scipy.special.psi(self._gamma[[doc_id], :]).T, (1, len(paths_lead_to_current_word)));
                    path_phi = numpy.tile((topic_sum + self._alpha).T, (1, len(paths_lead_to_current_word)));
                    assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                    
                    for path_index in xrange(len(paths_lead_to_current_word)):
                        path_phi[:, path_index] *= numpy.exp(numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1));
                    del path_index
                    
                    assert path_phi.shape==(self._number_of_topics, len(paths_lead_to_current_word));
                    # normalize path_phi over all topics
                    path_phi /= numpy.sum(path_phi);
                    
                    random_number = numpy.random.random();
                    for topic_index in xrange(self._number_of_topics):
                        for path_index in xrange(len(paths_lead_to_current_word)):
                            random_number-=path_phi[topic_index, path_index];
                            if random_number<=0:
                                break;
                        if random_number<=0:
                            break;
                    topic_sum[0, topic_index] += 1;
                    topic_path_assignment[word_index] = (topic_index, path_index);
                    
                    if sample_index >= self._burn_in_samples:
                        phi_sufficient_statistics[topic_index, paths_lead_to_current_word[path_index]] += 1;

                    '''
                    #phi_entropy += - numpy.sum((path_phi+1e-100) * numpy.log(path_phi+1e-100))
                    phi_entropy += - numpy.sum(path_phi * numpy.log(path_phi))
                    for path_index in xrange(len(paths_lead_to_current_word)):
                        phi_E_log_beta += numpy.sum(path_phi[:, [path_index]] * numpy.sum(self._E_log_beta[:, self._edges_along_path[paths_lead_to_current_word[path_index]]], axis=1)[:, numpy.newaxis])
                    del path_index
                    '''

                #del word_index, paths_lead_to_current_word
                
                #self._gamma[[doc_id], :] = self._alpha + topic_sum;
                
            self._gamma[[doc_id], :] = self._alpha + topic_sum;
            #gamma[doc_id] = self._alpha + topic_sum;
            
            #alpha_sufficient_statistics += compute_dirichlet_expectation(gamma[doc_id]);
            
            '''
            document_level_log_likelihood += scipy.special.gammaln(numpy.sum(self._alpha)) - numpy.sum(scipy.special.gammaln(self._alpha));
            document_level_log_likelihood += numpy.sum((self._alpha - 1) * compute_dirichlet_expectation(self._gamma[[doc_id], :]));

            document_level_log_likelihood += numpy.sum(topic_sum / len(self._data[doc_id]) * compute_dirichlet_expectation(self._gamma[[doc_id], :]));
            #document_level_log_likelihood += numpy.sum(numpy.sum(document_phi, axis=1)[:, numpy.newaxis].T * compute_dirichlet_expectation(self._gamma[[doc_id], :]));
            
            document_level_log_likelihood += phi_E_log_beta;
            
            document_level_log_likelihood += - scipy.special.gammaln(numpy.sum(self._gamma[[doc_id], :])) + numpy.sum(scipy.special.gammaln(self._gamma[[doc_id], :]))
            document_level_log_likelihood += - numpy.sum((self._gamma[[doc_id], :] - 1) * compute_dirichlet_expectation(self._gamma[[doc_id], :]));

            document_level_log_likelihood += phi_entropy;
            '''
            
            #phi_sufficient_statistics += document_phi;
        
            if (doc_id+1) % 1000==0:
                print "successfully processed %d documents..." % (doc_id+1);
                
            del doc_id
         
        phi_sufficient_statistics /= (self._number_of_samples - self._burn_in_samples);
        assert phi_sufficient_statistics.shape==(self._number_of_topics, self._number_of_paths);
        
        #assert alpha_sufficient_statistics.shape==(1, self._number_of_topics);

        return phi_sufficient_statistics, document_level_log_likelihood #gamma, alpha_sufficient_statistics;

    """
    def m_step(self, phi_sufficient_statistics):
        assert phi_sufficient_statistics.shape==(self._number_of_topics, self._number_of_paths);

        var_beta = numpy.tile(self._edge_prior, (self._number_of_topics, 1))
        assert var_beta.shape==(self._number_of_topics, self._number_of_edges);
        #for internal_node_index in self._edges_from_internal_node:
            #edges_indices_list = self._edges_from_internal_node[internal_node_index];
            #var_beta[:, edges_indices_list] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_internal_node[internal_node_index]], axis=1)[:, numpy.newaxis];
        
        for edge_index in self._index_to_edge:
            #print var_beta[:, edge_index].shape, numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis].shape;
            var_beta[:, [edge_index]] += numpy.sum(phi_sufficient_statistics[:, self._paths_through_edge[edge_index]], axis=1)[:, numpy.newaxis];
        del edge_index
        assert(var_beta.shape == (self._number_of_topics, self._number_of_edges));

        #print "var_beta"
        #print var_beta
        #sys.exit()
        
        self._E_log_beta = var_beta;
        for internal_node_index in self._edges_from_internal_node:
            edge_index_list = self._edges_from_internal_node[internal_node_index];
            self._E_log_beta[:, edge_index_list] = compute_dirichlet_expectation(self._E_log_beta[:, edge_index_list]);
        del internal_node_index, edge_index_list;

        
        corpus_level_log_likelihood = 0;
        '''
        for internal_node_index in self._edges_from_internal_node:
            edges_indices_list = self._edges_from_internal_node[internal_node_index];
            corpus_level_log_likelihood += (scipy.special.gammaln(numpy.sum(self._edge_prior[:, edges_indices_list])) - numpy.sum(scipy.special.gammaln(self._edge_prior[:, edges_indices_list]))) * self._number_of_topics;
            corpus_level_log_likelihood += numpy.sum(numpy.dot((self._edge_prior[:, edges_indices_list] - 1), var_beta[:, edges_indices_list].T));
            
            corpus_level_log_likelihood += numpy.sum(-scipy.special.gammaln(numpy.sum(var_beta[:, edges_indices_list], axis=1)) + numpy.sum(scipy.special.gammaln(var_beta[:, edges_indices_list]), axis=1));
            corpus_level_log_likelihood += numpy.sum(-(var_beta[:, edges_indices_list]-1) * compute_dirichlet_expectation(var_beta[:, edges_indices_list]));
        '''
        
        # TODO: add in alpha updating
        # compute the sufficient statistics for alpha and update
        #alpha_sufficient_statistics = scipy.special.psi(self._gamma) - scipy.special.psi(numpy.sum(self._gamma, axis=1)[:, numpy.newaxis]);
        #alpha_sufficient_statistics = numpy.sum(alpha_sufficient_statistics, axis=0)[numpy.newaxis, :];
        #self.update_alpha(alpha_sufficient_statistics)
        
        return corpus_level_log_likelihood
    """

    """
    """
    '''
    def train(self):
        self._counter += 1;
        
        clock_e_step = time.time();        
        phi_sufficient_statistics, document_level_log_likelihood, gamma, alpha_sufficient_statistics = self.e_step();
        clock_e_step = time.time() - clock_e_step;
        
        clock_m_step = time.time();        
        corpus_level_log_likelihood = self.m_step(phi_sufficient_statistics, alpha_sufficient_statistics);
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

if __name__ == "__main__":
    raise NotImplementedError;
