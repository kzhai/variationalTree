#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint, codecs
import datetime, os;

import scipy.io;
import nltk;
import numpy;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

def main():
    input_file_path = sys.argv[1];
    output_file_path = sys.argv[2];
    
    input_stream = codecs.open(input_file_path, 'r', 'utf-8');
    output_stream = codecs.open(output_file_path, 'w', 'utf-8');
    output_stream.write("# document portion\n");
    current_id=0;
    for line in input_stream:
        content = line.split();
        
        if int(content[0])>current_id+1:
            for index in xrange(current_id+1, int(content[0])):
                print "document %d collapsed..." % index;
                output_stream.write("%d null-source " % (index-1));
                output_stream.write(" ".join("%d 0.1" % item for item in xrange(10)));
                output_stream.write("\n");
        
        freqdist = nltk.probability.FreqDist();
        for index in xrange(1, len(content)):
            freqdist.inc(index-1, float(content[index]));
        
        output_stream.write(str(int(content[0])-1) + " null-source");
        for item in freqdist.keys():
            output_stream.write(" %d %f" % (item, freqdist[item]/freqdist.N()));
        output_stream.write("\n");
        
        current_id=int(content[0]);

if __name__ == '__main__':
    main()