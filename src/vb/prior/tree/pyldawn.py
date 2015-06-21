from pylda import *
from priortree import *

INT_BITS = 31

#class Path:
#    def __init__(self, word, indices):
#        self._nodes = indices
#        self.word = word
#        self._relatives = set()
#
#    def add_relative(self, path_id, word):
#        self._relatives.add((word, path_id))

class TopicWalk:
    def __init__(self, nodes):
        self._counts = defaultdict(FreqDist)
        self._node_counts = defaultdict()

    def change_count(self, path, increment):
        """
        For each node in the graph, change the count
        """
        for pp in range(0, len(path)-1):
            parent = path[pp]
            child = path[pp+1]
            self._counts[parent].inc(child, increment)

        affected_nodes = set()
        for pp in range(0, len(path)):
            node  = path[pp]
            if node not in self._node_counts.keys():
                self._node_counts[node] = 0

            old_count = self._node_counts[node]
            self._node_counts[node] += increment
            new_count = self._node_counts[node]

            if (pp != 0) and (old_count == 0 or new_count == 0):
                affected_nodes.add(node)

        return affected_nodes

    def probability(self, beta_lookup, path, prior_prob = 0.0):
        """
        Compute the probability of a path given the beta lookup (which provides
        smoothing).  Subtract out the prior probability of the path.
        """

        raise NotImplementedError

class WalkTopicCounts(LdaTopicCounts):
    def __init__(self):

        self._num_topics = 0
        self._max_depth = 0

        # Store beta for each transition
        self._beta = defaultdict(dict)

        # Store beta sum for each node
        self._beta_sum = {}

        # Store sum over smoothing only walks for each word
        self._prior_sum = {}
        self._prior_path_prob = defaultdict(dict)

        # path for each word
        self._word_paths = defaultdict(dict)

        # For each word, keep track of the topic and paths with non-zero
        # non-smoothing contributions
        self._non_zero_paths = defaultdict(dict)
        self._normalizer = defaultdict(dict)
        self._total_paths = 0

        # Track the number of transitions per topic per node per child
        self._traversals = defaultdict(TopicWalk)

        # Node to Word path index, so once a node is changed, easy to get the affected nodes
        self._node_pathindex = defaultdict()
        self._pathindex_word = []
        

    def add_path(self, vocab):
        """
        Add a path to the tree, record beta values (if beta values already
        exist, make sure they're consisting), add contribution to the prior_sum,
        and add an element to the words prior path prob.

        Initialize the traversals so that there is a FreqDist for each node in
        the path in each topic.
        """

        raise NotImplementedError

    def initialize_params(self, tree, vocab, topic_num):

        self._max_depth = tree._max_depth	
        self._num_topics = topic_num

		# this datastructure is to handle the case that one node contains multiple words
        leaf_index = len(tree._nodes.keys()) - 1
        word_leaf = defaultdict(dict)

        # initialize self._beta and self._beta_sum
        for index in tree._nodes.keys():
            node = tree._nodes[index]
            if len(node._children_offsets) > 0:
                assert len(node._words) == 0
                self._beta_sum[index] = node._transition_scalor
                for ii in range(0, len(node._children_offsets)):
                    child_index = node._children_offsets[ii]
                    self._beta[index][child_index] = node._transition_prior[ii]

            # leaf nodes might contain multiple words
            # if yes, set the prior according to the words count in this node
            # It is equal to changing a node containing multiple words to a node
            # containing multiple leaf node and each node contains only one word
            if len(node._words) > 1:
                assert len(node._children_offsets) == 0
                assert len(node._words) > 0
                self._beta_sum[index] = node._transition_scalor
                for ii in range(0, len(node._words)):
                    word_index = node._words[ii]
                    leaf_index += 1
                    word_leaf[word_index][index] = leaf_index
                    self._beta[index][leaf_index] = node._transition_prior[ii]

        # initialize self._prior_sum, self._prior_path_prob, self._word_paths
        # self._total_paths, self._node_pathindex
        path_index = -1
        for word in tree._word_paths.keys():
            self._prior_sum[word] = 0
            #print tree._word_paths[word]
            for pp in range(0, len(tree._word_paths[word])):
                path_index += 1
                self._total_paths += 1

                self._pathindex_word.append(word)

                prob = 1.0
                word_path = tree._word_paths[word][pp]
                word_path_node = word_path._nodes
                # this datastructure is to handle the case that 
                # one node contains multiple words
                # if yes, add a leaf index for each word
                parent = word_path_node[-1]
                if parent in word_leaf[word].keys():
                    leaf_index = word_leaf[word][parent]
                    word_path_node.append(leaf_index)

                for node_index in range(0, len(word_path_node) - 1):
                    parent = word_path_node[node_index]
                    child = word_path_node[node_index + 1]
                    prob *= self._beta[parent][child]

                for node_index in word_path_node:
                    if node_index not in self._node_pathindex.keys():
                        self._node_pathindex[node_index] = set()
                    self._node_pathindex[node_index].add(path_index)
          
                self._prior_path_prob[word][path_index] = prob
                self._prior_sum[word] += prob
                self._word_paths[word][path_index] = word_path_node


        # initialize self._non_zeros_paths and self._traversals
        for tt in range(0, topic_num):
            self._non_zero_paths[tt] = dict()
            for ww in tree._word_paths.keys():
                self._non_zero_paths[tt][ww] = FreqDist()

            tw = TopicWalk(tree._nodes)
            self._traversals[tt] = tw
        
    def set_vocabulary(self, words, default_beta = 0.01):
        """
        Use tree-based beta rather than flat
        """

        raise NotImplementedError

    def paths(self, term):
        """
        Return all the paths associated with a word
        """
        return self._word_paths[term]

    def sample_path_from_prior(self, term, rand_stub=None):
        """
        Sample a path from the prior distribution
        """
        sampled_path = -1
        if rand_stub:
            sample = rand_stub
        else:
            sample = random()
        sample *= self._prior_sum[term]
        for pp in self._prior_path_prob[term].keys():
            sample -= self._prior_path_prob[term][pp]
            if sample <= 0.0:
                sampled_path = pp
                break
        assert sampled_path >= 0
        return sampled_path

    def change_prior(self, word, beta):
        """
        This should not be called for this model.  Consider removing from
        interface?
        """
        raise NotImplementedError

    def initialize(self, word, topic):
        """
        Assign the topic (without decrementing) and sample a path from the prior
        distribution.
        """
        # sample a path       
        pp = self.sample_path_from_prior(word)

        # add to traversals
        tw = self._traversals[topic]
        sampled_path = self._word_paths[word][pp]
        tw.change_count(sampled_path, 1)

        # note we didn't initialize self._non_zero_paths,
        # till we finished loading all documents
        # we update the self._non_zero_paths in update_params()

        return pp


    def update_pathmasked_count(self, affected_paths, topic):
        tw = self._traversals[topic]
        for pp in affected_paths:
            # change path mask: only the edge mask, not the path count
            ww = self._pathindex_word[pp]
            path_nodes = self._word_paths[ww][pp]
            leaf_node = path_nodes[len(path_nodes) - 1]
            if leaf_node in self._traversals[topic]._node_counts.keys():
                original_count = self._traversals[topic]._node_counts[leaf_node]
            else:
                original_count = 0

            # note the root node is not included here
            internal_nodes = path_nodes[1 : len(path_nodes)]
            shift_count = INT_BITS
            count = self._max_depth - 1
            #count = 5
            val = 0.0
            flag = False
            for nn in internal_nodes:
                shift_count -= 1
                count -= 1
                if nn in tw._node_counts.keys() and tw._node_counts[nn] > 0:
                    flag = True
                    val += 1 << shift_count

            while flag and count > 0:
                shift_count -= 1
                val += 1 << shift_count
                count -= 1

            if ww not in self._non_zero_paths[topic].keys():
                self._non_zero_paths[topic][ww] = FreqDist()

            if (val+original_count) != 0.0:
                self._non_zero_paths[topic][ww][pp] = val + original_count
            elif pp in self._non_zero_paths[topic][ww].keys():
                self._non_zero_paths[topic][ww].pop(pp)

            
            #print original_count, self._non_zero_paths[topic][ww][pp]


    def update_params(self):
        # after load in all the documents, update related params
        paths = range(0, len(self._pathindex_word))
        for tt in range(0, self._num_topics):
            # update self._non_zero_paths
            self.update_pathmasked_count(paths, tt)
            # update self._normlizer
            self.compute_normalizer(tt)
            

    def compute_normalizer_path(self, topic, word, path_index):
        tw = self._traversals[topic]
        path_nodes = self._word_paths[word][path_index]
        internal_nodes = path_nodes[0:len(path_nodes)-1]
        
        norm = 1.0
        for node_index in internal_nodes:
            tmp = self._beta_sum[node_index]
            if node_index in tw._node_counts.keys():
               	tmp += tw._node_counts[node_index]
            norm *= tmp

        return norm


    def compute_normalizer(self, topic):
        
        for ww in self._word_paths.keys():
            for pp in self._word_paths[ww].keys():
                self._normalizer[topic][pp] = self.compute_normalizer_path(topic, ww, pp)


    def get_normalizer(self, topic):
        """
        Return the normalizer for all paths (including priors).
        """

        return self._normalizer[topic]


    def find_affected_paths(self, nodes):

        affected_paths = set()
        for node_index in nodes:
            for pp in self._node_pathindex[node_index]:
                affected_paths.add(pp)

        return affected_paths

    def change_count(self, topic, word, path_index, delta):
        """
        Change the count associated with a path.  This requires:
        1. Update the count in the topic walk associated with the topic
        2. Update the listing of affected paths that now have non-zero paths
        3. Update the cached normalizer for a topic
        """
        path_nodes = self._word_paths[word][path_index]
        tw = self._traversals[topic]

        # for affected paths, firstly remove the old values
        internal_nodes = path_nodes[0:len(path_nodes)-1]
        for node_index in internal_nodes:
            tmp = self._beta_sum[node_index]
            if node_index in tw._node_counts.keys():
               	tmp += tw._node_counts[node_index]
            for pp in self._node_pathindex[node_index]:
                self._normalizer[topic][pp] /= tmp

        # change the count for each edge, per topic
        # return the node index whose count changed: 0-->n or n-->0
        affected_nodes = tw.change_count(path_nodes, delta)
        # change path count
        self._non_zero_paths[topic][word].inc(path_index, delta)

        # if necessary, change the path mask of the affected nodes
        # find affected paths
        if len(affected_nodes) > 0:
            affected_paths = self.find_affected_paths(affected_nodes)
            self.update_pathmasked_count(affected_paths, topic)

        # update the cached normalizer for a topic
        # for affected paths, after changing count, recompute the new values
        for node_index in internal_nodes:
            tmp = self._beta_sum[node_index]
            if node_index in tw._node_counts.keys():
               	tmp += tw._node_counts[node_index]
            for pp in self._node_pathindex[node_index]:
                self._normalizer[topic][pp] *= tmp

    def get_prior(self, word):
        """
        Return the prior probability of a word marginalized over all paths
        """
        return self._prior_sum[word]

    def get_path_prior(self, word, path_index):
        return self._prior_path_prob[word][path_index]

    def get_observations(self, topic, word, path_index):
        """
        Return the counts of a word/path combination.
        """
        path_nodes = self._word_paths[word][path_index]
        val = 1.0
        tw = self._traversals[topic]
        for index in range(0, len(path_nodes)-1):
            parent = path_nodes[index]
            child = path_nodes[index+1]
            tmp = self._beta[parent][child] + tw._counts[parent][child]
            val *= tmp
        val -= self._prior_path_prob[word][path_index]
        return val

    def report(self, vocab, outputfilename, limit):
        """
        Print out a human readable report
        """
        outputfile = open(outputfilename, 'w')

        for tt in self._normalizer.keys():
            #count = sum(self._non_zero_paths[tt].values())
            #print("---------\nTopic %i (%i tokens)\n-------------" %\
            #      (tt, count))
            #print("---------\nTopic %i\n-------------" % tt)
            outputfile.write("\n---------\nTopic %i\n-------------\n" % tt)
            normalizer = self.get_normalizer(tt)
            top_words = FreqDist()
            for ww in self._non_zero_paths[tt].keys():
                for pp in self._non_zero_paths[tt][ww].keys():
                    #print "normalizer: ", tt, pp, normalizer[pp]
                    val = self.get_observations(tt, ww, pp) + self._prior_path_prob[ww][pp]
                    #val = self.get_observations(tt, ww, pp)
                    val /= normalizer[pp]
                    top_words[(ww, pp)] = val

            word = 0
            for (ww, pp) in top_words.keys():
                val = top_words[(ww, pp)]
                #print("%0.5f\t%s" % (val, vocab[ww]))
                outputfile.write("%0.5f\t%s\n" % (val, vocab[ww]))
                word += 1
                if word > limit:
                    break
        outputfile.close()

    def compute_topic_terms(self, alpha, doc_counts, term):
        """
        Iterate over all the paths that have a non-zero non-smoothing
        contribution (as stored in prior_path_prob)
        """
        d = {}
        for tt in range(0, self._num_topics):
            normalizer = self.get_normalizer(tt)
            for pp in self._non_zero_paths[tt][term].keys():
                obs = self.get_observations(tt, term, pp)
                if obs == 0:
                    break
                val = (alpha[tt] + doc_counts[tt]) * obs
                val /= normalizer[pp]
                d[(tt, pp)] = val
                assert d[(tt, pp)] >= 0
        return d

class ldawnSampler(Sampler):
    def __init__(self, num_topics, vocab, tree_files, hyper_file, topics=None, alpha=0.1):

        self._num_topics = num_topics
        self._vocab = vocab

        self._doc_counts = defaultdict(FreqDist)
        self._doc_tokens = defaultdict(list)
        self._doc_assign = defaultdict(list)
        self._doc_path_assign = defaultdict(list)
        self._alpha = [alpha for x in xrange(num_topics)]
        self._sample_stats = defaultdict(int)

        # initialize tree
        self._tree = PriorTree()
        self._tree.initialize(tree_files, hyper_file, vocab)

        # initialize topics
        self._topics = WalkTopicCounts()
        self._topics.initialize_params(self._tree, self._vocab, self._num_topics)

        # self._smoothing_only_mass
        self._smoothing_only_mass = defaultdict()
        self._topic_beta_mass = defaultdict()

        self._lhood = []
        self._time = []


    def update_params(self):
        # after load in all documents, update related params

        # update the related params in self._topics
        self._topics.update_params()

        # update smoothing bucket
        self.compute_smoothing_only()


    def compute_smoothing_only(self):
        """
        Different from Yao, normalizer and prior for each path is different
        """
        for tt in xrange(self._num_topics):
            topics = self._topics
            normalizer = self._topics.get_normalizer(tt)
            for ww in self._topics._word_paths.keys():
                if ww not in self._smoothing_only_mass.keys():
                    self._smoothing_only_mass[ww] = 0
                for pp in self._topics._word_paths[ww].keys():
                    val = self._alpha[tt] * topics.get_path_prior(ww, pp) / normalizer[pp]
                    self._smoothing_only_mass[ww] += val


    def compute_topic_beta(self, doc_counts):
        """
        normalizer and prior for each path is different
        """
        self._topic_beta_mass = defaultdict()
        for tt in doc_counts:
            topics = self._topics
            normalizer = self._topics.get_normalizer(tt)
            for ww in self._topics._word_paths.keys():
                if ww not in self._topic_beta_mass.keys():
                    self._topic_beta_mass[ww] = 0
                for pp in self._topics._word_paths[ww].keys():
                    val = doc_counts[tt] * topics.get_path_prior(ww, pp) / normalizer[pp]
                    self._topic_beta_mass[ww] += val



    def compute_term_smoothing_only(self, term):
        """
        Different from Yao, normalizer and prior for each word is different
        """
        smoothing = 0.0
        for tt in xrange(self._num_topics):
            topics = self._topics
            normalizer = topics.get_normalizer(tt)
            for pp in topics._word_paths[term].keys():
                val = self._alpha[tt] * topics.get_path_prior(term, pp) / normalizer[pp]
                smoothing += val
        assert smoothing > 0.0
        return smoothing


    def compute_term_topic_beta(self, doc_counts, term):
        """
        normalizer and prior for each word is different
        """
        topic_beta = 0.0

        tmp = 0
        for tt in doc_counts:
            topics = self._topics
            normalizer = self._topics.get_normalizer(tt)
            if doc_counts[tt] > 0:
                tmp += 1
                for pp in self._topics._word_paths[term].keys():
                    val = doc_counts[tt] * topics.get_path_prior(term, pp) / normalizer[pp]
                    topic_beta += val

        if tmp == 0:
            assert topic_beta == 0.0
        else:
            assert topic_beta > 0.0, "Topic beta falls to %s!" % topic_beta

        return topic_beta


    def getAffectedWords(self, term, path_index):
        affected = defaultdict(set)
        path_nodes = self._topics._word_paths[term][path_index]
        internal_nodes = path_nodes[0:len(path_nodes)-1]
        affected = defaultdict(set)
        for node_index in internal_nodes:
            for pp in self._topics._node_pathindex[node_index]:
                ww = self._topics._pathindex_word[pp]
                affected[ww].add(pp)
        return affected


    def change_topic(self, doc, index, term, new_topic, new_path):
        """
        Change the topic of a token in a document.  Update the counts
        appropriately.
        """

        alpha = self._alpha
        assert index < len(self._doc_assign[doc]), \
               "Bad index %i for document %i, term %i %s" % \
               (index, doc, term, str(self._doc_assign[doc]))
        old_topic = self._doc_assign[doc][index]
        old_path = self._doc_path_assign[doc][index]
        
        if old_topic != -1:
            assert new_topic == -1

            self._topics.change_count(old_topic, term, old_path, -1)
            self._doc_counts[doc].inc(old_topic, -1)
            self._doc_assign[doc][index] = -1
            self._doc_path_assign[doc][index] = -1

        if new_topic != -1:
            assert old_topic == -1

            self._topics.change_count(new_topic, term, new_path, +1)
            self._doc_counts[doc].inc(new_topic, +1)
            self._doc_assign[doc][index] = new_topic
            self._doc_path_assign[doc][index] = new_path

    def change_topic_correct(self, doc, index, term, new_topic, new_path):
        """
        Change the topic of a token in a document.  Update the counts
        appropriately.
        """

        alpha = self._alpha
        assert index < len(self._doc_assign[doc]), \
               "Bad index %i for document %i, term %i %s" % \
               (index, doc, term, str(self._doc_assign[doc]))
        old_topic = self._doc_assign[doc][index]
        old_path = self._doc_path_assign[doc][index]
        
        if old_topic != -1:
            assert new_topic == -1

            normalizer = self._topics.get_normalizer(old_topic)
            # for affected paths of words, firstly remove the old values
            affected = self.getAffectedWords(term, old_path)
            #print affected
            assert term in affected.keys()
            assert old_path in affected[term]
            for ww in affected.keys():
                for pp in affected[ww]:
                    tmp = self._topics.get_path_prior(ww, pp) / normalizer[pp]
                    val = self._alpha[old_topic] * tmp
                    self._smoothing_only_mass[ww] -= val
                    val = self._doc_counts[doc][old_topic] * tmp
                    self._topic_beta_mass[ww] -= val

            self._topics.change_count(old_topic, term, old_path, -1)
            self._doc_counts[doc].inc(old_topic, -1)
            self._doc_assign[doc][index] = -1
            self._doc_path_assign[doc][index] = -1

            # Add to weights
            normalizer = self._topics.get_normalizer(old_topic)
            # for affected paths of words, add the new values back
            for ww in affected.keys():
                for pp in affected[ww]:
                    tmp = self._topics.get_path_prior(ww, pp) / normalizer[pp]
                    val = self._alpha[old_topic] * tmp
                    self._smoothing_only_mass[ww] += val
                    val = self._doc_counts[doc][old_topic] * tmp
                    self._topic_beta_mass[ww] += val

            #print "test: ", tmp_original2, tmp_val2, self._smoothing_only_mass[term], normalizer[old_path]

            assert self._smoothing_only_mass[term] > 0.0, "Smoothing only fell to %f" % \
                self._smoothing_only_mass[term]

            assert self._topic_beta_mass[term] > 0.0, "Topic beta fell to %f" % \
                self._topic_beta_mass[term]

        if new_topic != -1:
            assert old_topic == -1
            normalizer = self._topics.get_normalizer(new_topic)
            # for affected paths of words, firstly remove the old values
            affected = self.getAffectedWords(term, new_path)
            assert term in affected.keys()
            assert new_path in affected[term]
            for ww in affected.keys():
                for pp in affected[ww]:
                    tmp = self._topics.get_path_prior(ww, pp) / normalizer[pp]
                    val = self._alpha[new_topic] * tmp
                    self._smoothing_only_mass[ww] -= val
                    val = self._doc_counts[doc][new_topic] * tmp
                    self._topic_beta_mass[ww] -= val

            self._topics.change_count(new_topic, term, new_path, +1)
            self._doc_counts[doc].inc(new_topic, +1)
            self._doc_assign[doc][index] = new_topic
            self._doc_path_assign[doc][index] = new_path

            normalizer = self._topics.get_normalizer(new_topic)
            for ww in affected.keys():
                for pp in affected[ww]:
                    tmp = self._topics.get_path_prior(ww, pp) / normalizer[pp]
                    val = self._alpha[new_topic] * tmp
                    self._smoothing_only_mass[ww] += val
                    val = self._doc_counts[doc][new_topic] * tmp
                    self._topic_beta_mass[ww] += val

            #print "test: ", tmp_original2, tmp_val2, self._smoothing_only_mass[term], normalizer[new_path]

            assert self._smoothing_only_mass[term] > 0.0, "Smoothing only fell to %f" % \
                self._smoothing_only_mass[term]

            assert self._topic_beta_mass[term] > 0.0, "Topic beta fell to %f" % \
                self._topic_beta_mass[term]


    def change_topic_old(self, doc, index, term, new_topic, new_path):
        """
        Change the topic of a token in a document.  Update the counts
        appropriately.
        """

        alpha = self._alpha
        assert index < len(self._doc_assign[doc]), \
               "Bad index %i for document %i, term %i %s" % \
               (index, doc, term, str(self._doc_assign[doc]))
        old_topic = self._doc_assign[doc][index]
        old_path = self._doc_path_assign[doc][index]
        
        if old_topic != -1:
            assert new_topic == -1
            tmp_original1 = self._smoothing_only_mass[term]
            # Subtract out contribution to weights
            normalizer = self._topics.get_normalizer(old_topic)
            #tmp = self._topics.get_path_prior(term, old_path) / normalizer[old_path]
            #val = self._alpha[old_topic] * tmp
            #tmp_val1 = val
            #self._smoothing_only_mass[term] -= val
            #val = self._doc_counts[doc][old_topic] * tmp
            #self._topic_beta_mass[term] -= val

            # for affected paths of words, firstly remove the old values
            affected = self.getAffectedWords(old_path)
            assert term in affected.keys()
            assert old_path in affected[term]
            for ww in affected.keys():
                for pp in affected[ww]:
                    tmp = self._topics.get_path_prior(ww, pp) / normalizer[pp]
                    val = self._alpha[old_topic] * tmp
                    self._smoothing_only_mass[ww] -= val
                    val = self._doc_counts[doc][old_topic] * tmp
                    self._topic_beta_mass[ww] -= val
                    


            #if tmp_original1 < tmp_val1:
            #    print "info: ", doc, index, term, old_topic, new_topic
            #    print "test: ", tmp_original1, tmp_val1, self._smoothing_only_mass[term], normalizer[old_path]
            #    test_initialization(self)
            #    assert 1 < 0

            self._topics.change_count(old_topic, term, old_path, -1)
            self._doc_counts[doc].inc(old_topic, -1)
            self._doc_assign[doc][index] = -1
            self._doc_path_assign[doc][index] = -1

            # Add to weights
            #tmp_original2 = self._smoothing_only_mass[term]
            normalizer = self._topics.get_normalizer(old_topic)
            #tmp = self._topics.get_path_prior(term, old_path) / normalizer[old_path]
            #val = self._alpha[old_topic] * tmp
            #tmp_val2 = val
            #self._smoothing_only_mass[term] += val
            #val = self._doc_counts[doc][old_topic] * tmp
            #self._topic_beta_mass[term] += val

            for ww in affected.keys():
                for pp in affected[ww]:
                    tmp = self._topics.get_path_prior(ww, pp) / normalizer[pp]
                    val = self._alpha[old_topic] * tmp
                    self._smoothing_only_mass[ww] += val
                    val = self._doc_counts[doc][old_topic] * tmp
                    self._topic_beta_mass[ww] += val

            #print "test: ", tmp_original2, tmp_val2, self._smoothing_only_mass[term], normalizer[old_path]

            assert self._smoothing_only_mass[term] > 0.0, "Smoothing only fell to %f" % \
                self._smoothing_only_mass[term]

            assert self._topic_beta_mass[term] > 0.0, "Topic beta fell to %f" % \
                self._topic_beta_mass[term]

        if new_topic != -1:
            assert old_topic == -1
            tmp_original1 = self._smoothing_only_mass[term]
            normalizer = self._topics.get_normalizer(new_topic)
            tmp = self._topics.get_path_prior(term, new_path) / normalizer[new_path]
            val = self._alpha[new_topic] * tmp
            tmp_val1 = val
            self._smoothing_only_mass[term] -= val
            val = self._doc_counts[doc][new_topic] * tmp
            self._topic_beta_mass[term] -= val

            if tmp_original1 < tmp_val1:
                print ""
                print "info: ", doc, index, term, old_topic, new_topic
                print "test: ", tmp_original1, tmp_val1, self._smoothing_only_mass[term], normalizer[new_path]
                print ""
                test_initialization(self)
                assert 1 < 0


            self._topics.change_count(new_topic, term, new_path, +1)
            self._doc_counts[doc].inc(new_topic, +1)
            self._doc_assign[doc][index] = new_topic
            self._doc_path_assign[doc][index] = new_path

            tmp_original2 = self._smoothing_only_mass[term]
            normalizer = self._topics.get_normalizer(new_topic)
            tmp = self._topics.get_path_prior(term, new_path) / normalizer[new_path]
            val = self._alpha[new_topic] * tmp
            tmp_val2 = val
            self._smoothing_only_mass[term] += val
            val = self._doc_counts[doc][new_topic] * tmp
            self._topic_beta_mass[term] += val

            #print "test: ", tmp_original2, tmp_val2, self._smoothing_only_mass[term], normalizer[new_path]

            assert self._smoothing_only_mass[term] > 0.0, "Smoothing only fell to %f" % \
                self._smoothing_only_mass[term]

            assert self._topic_beta_mass[term] > 0.0, "Topic beta fell to %f" % \
                self._topic_beta_mass[term]


    def run_sampler(self, iterations = 100):

        '''
        print '-------------'
        for tt in range(0, self._num_topics):
            for ii in self._topics._traversals[tt]._counts.keys():
                for jj in self._topics._traversals[tt]._counts[ii].keys():
                    print 'topic', tt, ':', ii, jj, self._topics._traversals[tt]._counts[ii][jj]
        print '------------'

        print self._topics.get_normalizer(0)
        print self._topics.get_normalizer(1)
        print self._topics.get_normalizer(2)

        for doc_id in self._doc_tokens.keys():
            print "-------"
            print doc_id
            print "-------"
            print self._doc_tokens[doc_id]
            print self._doc_counts[doc_id]
            print self._doc_assign[doc_id]
        '''

        self.update_params()

        average_time = 0.0
        print "Start sampling!"
        for ii in xrange(iterations):
            #if ii % 20 == 0:
            #    print("Iteration %i" % ii)
            start = time.time()
            doc_count = 0
            for jj in self._doc_assign:
                doc_count += 1
                if doc_count % 100 == 0:
                    print("Sampled %i documents" % doc_count)
                #self.sample_doc(jj, rand_stub=0.5)
                self.sample_doc(jj)
            total = time.time() - start
            lhood = self.lhood()
            print("Iteration %i, likelihood %f, %0.5f seconds" % (ii, lhood, total))
            self._lhood.append(lhood)
            self._time.append(total)
            average_time += total

        print "Finish sampling! The average time for each iteration is %0.5f seconds" % (average_time/iterations)


    def sample_doc(self, doc_id, rand_stub=None, debug=False):
        """
        For a single document, compute the conditional probabilities and
        resample topic assignments.
        """

        one_doc_topics = self._doc_assign[doc_id]
        local_topic_counts = self._doc_counts[doc_id]
        num_topics = self._num_topics
        alpha = self._alpha
        
        topics = self._topics

        # compute topic_beta bucket for local document
        #self.compute_topic_beta(local_topic_counts)
        
        for index in xrange(len(one_doc_topics)):
            term = self._doc_tokens[doc_id][index]
            path_index = self._doc_path_assign[doc_id][index]

            self.change_topic(doc_id, index, term, -1, -1)

            #smoothing_only_mass = self._smoothing_only_mass[term]
            #topic_beta_mass = self._topic_beta_mass[term]
            local_topic_counts = self._doc_counts[doc_id]
            smoothing_only_mass = self.compute_term_smoothing_only(term)

            topic_beta_mass = self.compute_term_topic_beta(local_topic_counts, term)

            topic_term_scores = topics.compute_topic_terms(alpha,
                                                           local_topic_counts,
                                                           term)
            topic_term_mass = sum(topic_term_scores.values())

            norm = (smoothing_only_mass + topic_beta_mass + topic_term_mass)
            assert norm > 0, "Normalizer %f" % norm

            if debug:                
                print("Sm Only: %f" % (smoothing_only_mass))
                print("TopBet: %f" % (topic_beta_mass))
                print("TT: %s" % (topic_term_mass))
                print("Norm: %f" % norm)

            if rand_stub:
                sample = rand_stub
            else:
                sample = random()

            sample *= norm
            original_sample = sample

            if debug:
                probs = [smoothing_only_mass, topic_beta_mass, topic_term_mass]
                probs = ["%i:%0.4f" % (x, (probs[x] / sum(probs))) for x \
                                           in xrange(len(probs))]
                print(["Sample:%0.3f" % sample] + probs)

            new_topic = -1
            new_path = -1

            if sample < smoothing_only_mass:
                if debug:
                    print("Index %i Smoothing Only" % index)

                self._sample_stats["smoothing_only"] += 1

                for kk in range(0, self._num_topics):
                    normalizer = topics.get_normalizer(kk)
                    for pp in topics._word_paths[term].keys():
                        val = alpha[kk] * topics.get_path_prior(term, pp) 
                        val /= normalizer[pp]
                        sample -= val
                        if sample <= 0.0:
                            new_topic = kk
                            new_path = pp
                            break  
                    if new_topic >= 0:
                        break

                assert (new_topic >= 0 and new_topic < num_topics), \
                    "Something went wrong in sampling smoothing only"
            else:
		    	sample -= smoothing_only_mass

            if new_topic < 0 and sample < topic_beta_mass:
                self._sample_stats["topic_beta"] += 1
                if debug:
                    print("Index %i Topic Beta" % index)

                for kk in local_topic_counts:
                    normalizer = topics.get_normalizer(kk)
                    for pp in topics._word_paths[term].keys():
                        val = local_topic_counts[kk] * topics.get_path_prior(term, pp) 
                        val /= normalizer[pp]
                        sample -= val
                        if sample <= 0.0:
                            new_topic = kk
                            new_path = pp
                            break
                    if new_topic >= 0:
                        break

                assert (new_topic >= 0 and new_topic < num_topics), \
                    "Something went wrong in sampling smoothing only"

            else:
                sample -= topic_beta_mass

            if new_topic < 0: 
                assert sample < topic_term_mass

                if debug:
                    print("Index %i Topic Term" % index)

                self._sample_stats["topic_term"] += 1
                #for kk in local_topic_counts:
                #    for pp in topics.paths(term):
                #        if debug:
                #            print("\tTopic %i Path %i: %0.4f" % \
                #                      (kk, pp, topic_term_scores[(kk, pp)]))
                #        
                #        sample -= topic_term_scores.get((kk, pp), 0.0)
                #        if sample <= 0.0:
                #            new_topic = kk
                #            new_path = pp
                #            break
                #    if new_topic >= 0:
                #        break
                for (kk, pp) in topic_term_scores.keys():
                    sample -= topic_term_scores.get((kk, pp), 0.0)
                    if sample <= 0.0:
                        new_topic = kk
                        new_path = pp
                        break
                assert new_topic >= 0, "Something went wrong in sampling topic term"
                assert new_topic < num_topics

            if debug:
                print("\t\t----> New topic: %i" % new_topic)

            self.change_topic(doc_id, index, term, new_topic, new_path)

        return self._doc_assign[doc_id]


    def topic_lhood_path(self):
        ##### not correct
        val = 0.0
        beta_sum_all = sum(self._topics._beta_sum.values())
        val += lgammln(beta_sum_all) * self._num_topics
        tmp = 0.0
        for nn in self._topics._beta_sum.keys():
            tmp += lgammln(self._topics._beta_sum[nn])
        val -= tmp * self._num_topics

        for tt in range(0, self._num_topics):
            normalizer = self._topics._normalizer[tt]
            total_count = 0
            for pp in normalizer.keys():
                ww = self._topics._pathindex_word[pp]
                path_nodes = self._topics._word_paths[ww][pp]

                node = path_nodes[len(path_nodes) - 1]
                if node in self._topics._traversals[tt]._node_counts.keys():
                    original_count = self._topics._traversals[tt]._node_counts[node]
                else:
                    original_count = 0

                val += lgammln(self._topics._prior_path_prob[ww][pp] + original_count)
                total_count += original_count
            val -= lgammln(total_count + beta_sum_all)

        return val

    def topic_lhood(self):

        topics = self._topics
        val = 0.0

        for tt in range(0, self._num_topics):

            for nn in topics._beta_sum.keys():
                beta_sum = topics._beta_sum[nn]
                val += lgammln(beta_sum) * len(topics._beta[nn])

                tmp = 0.0
                for cc in topics._beta[nn].keys():
                    tmp += lgammln(topics._beta[nn][cc])
                val -= tmp * len(topics._beta[nn])

                for cc in topics._beta[nn].keys():
                    original_count = topics._traversals[tt]._counts[nn][cc]
                    val += lgammln(topics._beta[nn][cc] + original_count)

                if nn in topics._traversals[tt]._node_counts.keys():
                    val -= lgammln(topics._traversals[tt]._node_counts[nn] + beta_sum)
                else:
                    val -= lgammln(beta_sum)
            #print "likelihood ", val

        return val

        

