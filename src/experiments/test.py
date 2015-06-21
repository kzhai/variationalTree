#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint, codecs
import datetime, os;

import scipy.io;
import nltk;
import numpy;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def main():
    test_input_data = sys.argv[1];
    test_output_file = sys.argv[2];
    model_directory = sys.argv[3];
    model_vocab_file = sys.argv[4];
    
    #if not os.path.exists(test_output_file):
        #os.mkdir(test_output_file);
    from vb.prior.tree.hybrid import Hybrid, parse_data
    
    data, type_to_index, index_to_type, vocabulary = parse_data(test_input_data, model_vocab_file);

    hybrid_inferencer = Hybrid();
    hybrid_inferencer.load_params(os.path.join(model_directory, 'current-params'));
    hybrid_inferencer.load_tree(os.path.join(model_directory, 'current-tree'));
    hybrid_inferencer.load_E_log_beta(os.path.join(model_directory, 'current-E-log-beta'));
    
    #gamma = hybrid_inferencer.test(data);
    
    #output_stream = codecs.open(os.path.join(test_output_file, 'gamma'), 'w', 'utf-8');
    output_stream = codecs.open(test_output_file, 'w', 'utf-8');
    output_stream.write("# " + test_output_file + "\n");

    '''
    for document_id in xrange(gamma.shape[0]):
        gamma[[document_id], :] /= numpy.sum(gamma[[document_id], :]);
    
        freqdist = nltk.probability.FreqDist();
        for topic_index in xrange(hybrid_inferencer._number_of_topics):
            freqdist.inc(topic_index, gamma[document_id, topic_index]);
         
        output_stream.write("%d null-source" % document_id); 
        for item in freqdist.keys():
            output_stream.write(" %d %f" %(item, gamma[document_id, item]));
        output_stream.write("\n");
    '''

    for document_id in xrange(len(data)):
        gamma = hybrid_inferencer.test([data[document_id]]);
        gamma /= numpy.sum(gamma);
    
        freqdist = nltk.probability.FreqDist();
        for topic_index in xrange(hybrid_inferencer._number_of_topics):
            freqdist.inc(topic_index, gamma[0, topic_index]);
         
        output_stream.write("%d null-source" % document_id); 
        for item in freqdist.keys():
            output_stream.write(" %d %f" %(item, gamma[0, item]));
        output_stream.write("\n");

        if document_id % 10000==0:
            print "processed %d documents" % document_id
        
        #output_stream.write("%d null-source %s\n" %(document_id, " ".join("%f" %item for item in gamma[document_id][0, :])));
    
if __name__ == '__main__':
    main()
