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

unwanted_chars = string.letters + string.digits + string.punctuation
#unwanted_chars += "!\"#\$%&'\(\)\*\+,-\./:;<=>\?@\[\\\]^_`\{|\}~"

print unwanted_chars
#unwanted_chars_regex = re.compile("[%s]" % unwanted_chars);
white_space_regex = re.compile(r'[\s]+');

def main():
    input_file_path = sys.argv[1];
    output_file_path = sys.argv[2];
    
    input_stream = codecs.open(input_file_path, 'r', 'utf-8');
    output_stream = codecs.open(output_file_path, 'w', 'utf-8');
    
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    remove_letters_map = dict((ord(char), None) for char in string.letters)
    remove_digits_map = dict((ord(char), None) for char in string.digits)
    
    line_counter = 0;
    for line in input_stream:
        line_counter += 1;
        content = line.strip();

        '''
        tokens = line.split("\t");
        if len(tokens)!=2:
            continue
        title = tokens[0];
        title_tokens = title.split();
        if len(title_tokens)>3:
            print line_counter, len(title_tokens)
            continue;
        title = title_tokens[0] + " " + " ".join(title_tokens[2:])
        '''

        content = content.translate(remove_punctuation_map);
        #content = content.translate(remove_letters_map);
        #content = content.translate(remove_digits_map);
        #content = re.sub(unwanted_chars_regex, " ", content);
        content = re.sub(white_space_regex, " ", content);

        if len(content)==0:
            print "warning: document '%s' collapsed..." % line

        output_stream.write("%s\n" % (content));

if __name__ == '__main__':
    main()
