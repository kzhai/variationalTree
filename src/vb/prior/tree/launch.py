#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import scipy.io;
import nltk;
import numpy;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def main():
    import option_parser;
    options = option_parser.parse_args();

    # parameter set 2
    assert(options.number_of_topics>0);
    number_of_topics = options.number_of_topics;
    assert(options.number_of_iterations>0);
    number_of_iterations = options.number_of_iterations;

    # parameter set 3
    alpha = 1.0/number_of_topics;
    if options.alpha>0:
        alpha=options.alpha;
    
    #assert options.default_correlation_prior>0;
    #default_correlation_prior = options.default_correlation_prior;
    #assert options.positive_correlation_prior>0;
    #positive_correlation_prior = options.positive_correlation_prior;
    #assert options.negative_correlation_prior>0;
    #negative_correlation_prior = options.negative_correlation_prior;
    
    # parameter set 4
    #disable_alpha_theta_update = options.disable_alpha_theta_update;
    hybrid_mode = options.hybrid_mode;
    update_hyperparameter = options.update_hyperparameter;
    
    # parameter set 5
    assert(options.snapshot_interval>0);
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    
    # parameter set 1
    assert(options.corpus_name!=None);
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    assert(options.tree_name!=None);

    corpus_name = options.corpus_name;

    input_directory = options.input_directory;
    if not input_directory.endswith('/'):
        input_directory += '/';
    input_directory += corpus_name+'/';
        
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    if not output_directory.endswith('/'):
        output_directory += '/';
    output_directory += corpus_name+'/';
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    tree_name = options.tree_name.strip();

    # create output directory
    now = datetime.datetime.now();
    output_directory += now.strftime("%y%b%d-%H%M%S")+"";
    output_directory += "-prior_tree-K%d-I%d-a%g-S%d-%s-%s-%s/" \
                        % (number_of_topics,
                           number_of_iterations,
                           alpha,
                           snapshot_interval,
                           tree_name,
                           hybrid_mode,
                           update_hyperparameter);

    #output_directory += "-prior_tree_uvb-K%d-I%d-a%g-dcp%g-pcp%g-ncp%g-S%d/" \
                        #% (number_of_topics,
                           #number_of_iterations,
                           #alpha,
                           #default_correlation_prior,
                           #positive_correlation_prior,
                           #negative_correlation_prior,
                           #snapshot_interval);

    os.mkdir(os.path.abspath(output_directory));

    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    options_output_file.write("tree_name=" + str(tree_name) + "\n");
    # parameter set 2
    options_output_file.write("number_of_iteration=%d\n" % (number_of_iterations));
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha=" + str(alpha) + "\n");
    #options_output_file.write("default_correlation_prior=" + str(default_correlation_prior) + "\n");
    #options_output_file.write("positive_correlation_prior=" + str(positive_correlation_prior) + "\n");
    #options_output_file.write("negative_correlation_prior=" + str(negative_correlation_prior) + "\n");
    # parameter set 4
    options_output_file.write("hybrid_mode=%s\n" % (hybrid_mode));
    options_output_file.write("update_hyperparameter=%s\n" % (update_hyperparameter));
    # parameter set 5
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");

    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    print "tree prior file=" + str(tree_name)
    # parameter set 2
    print "number_of_iterations=%d" %(number_of_iterations);
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha=" + str(alpha)
    #print "default_correlation_prior=" + str(default_correlation_prior)
    #print "positive_correlation_prior=" + str(positive_correlation_prior)
    #print "negative_correlation_prior=" + str(negative_correlation_prior)
    # parameter set 4
    print "hybrid_mode=%s" % (hybrid_mode)
    print "update_hyperparameter=%s" % (update_hyperparameter);
    # parameter set 5
    print "snapshot_interval=" + str(snapshot_interval);
    print "========== ========== ========== ========== =========="

    if hybrid_mode:
        import hybrid;
        lda_inference = hybrid.Hybrid(update_hyperparameter);
        import hybrid.parse_data as parse_data
    else:
        import uvb;
        lda_inference = uvb.UncollapsedVariationalBayes(update_hyperparameter);
        import uvb.parse_data as parse_data
        
    documents, type_to_index, index_to_type, vocabulary = parse_data(input_directory+'doc.dat', input_directory+'voc.dat');
    print "successfully load all training documents..."

    # initialize tree
    import priortree
    prior_tree = priortree.PriorTree();
    #from vb.prior.tree.priortree import PriorTree;
    #prior_tree = PriorTree();
    #prior_tree._initialize(input_directory+"tree.wn.*", vocabulary, default_correlation_prior, positive_correlation_prior, negative_correlation_prior);
    prior_tree.initialize(input_directory+tree_name+".wn.*", input_directory+tree_name+".hyperparams", vocabulary)

    lda_inference._initialize(documents, prior_tree, type_to_index, index_to_type, number_of_topics, alpha);
    
    for iteration in xrange(number_of_iterations):
        lda_inference.train();
        
        if (lda_inference._counter % snapshot_interval == 0):
            lda_inference.export_topic_term_distribution(output_directory + 'exp_beta-' + str(lda_inference._counter));
    
if __name__ == '__main__':
    main()