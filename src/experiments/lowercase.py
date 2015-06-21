#!/usr/bin/python
import cPickle, string, numpy, getopt, sys, random, time, re, pprint, codecs
import datetime, os;
import collections
import scipy.io;
import nltk;
import numpy;
import sre_constants;

from nltk.corpus import stopwords;
from nltk.probability import FreqDist;

white_space_regex = re.compile(r'[\s]+');

def main():
    input_file_path = sys.argv[1];
    output_file_path = sys.argv[2];
    
    input_stream = codecs.open(input_file_path, 'r', 'utf-8');
    output_stream = codecs.open(output_file_path, 'w', 'utf-8');
    
    line_counter = 0;
    for line in input_stream:
        line_counter += 1;
        content = line.strip();
        content = content.lower();
        content = re.sub(white_space_regex, " ", content);

        if len(content)==0:
            print "warning: document '%s' collapsed..." % line

        output_stream.write("%s\n" % (content));

if __name__ == '__main__':
    main()
