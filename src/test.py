import numpy;
import scipy;
import scipy.sparse
import scipy.io;
import random;
import itertools;
import nltk;
import time;
import util.log_math
import os;
import itertools;
import re;
import math;

try:
    import cPickle as pickle
except:
    import pickle

from collections import defaultdict;

"""
"""
def test():
    a = dict();
    for i in xrange(5):
        a[numpy.random.randint(1000)] = numpy.random.random();
    print a
    print a.keys();
    print a.values();
    print numpy.array([a.values()]);

"""
def measure_sparse_matrix_time():
    clock = time.time();
    for i in xrange(100):
        b = scipy.sparse.lil_matrix(numpy.round(numpy.random.random((2, 3))), dtype='uint8');
        b[1, :] = numpy.round(numpy.random.random(3));
    clock = time.time()-clock;
    print clock

    clock = time.time();
    for i in xrange(100):
        b = scipy.sparse.dok_matrix(numpy.round(numpy.random.random((2, 3))), dtype='uint8');
        b[1, :] = numpy.round(numpy.random.random(3));
    clock = time.time()-clock;
    print clock
    
    clock = time.time();
    for i in xrange(100):
        b = scipy.sparse.lil_matrix(numpy.round(numpy.random.random((2, 3))), dtype='uint8');
        c = scipy.sparse.vstack((b, b), format='lil', dtype='uint8');
    clock = time.time()-clock;
    print clock

    clock = time.time();
    for i in xrange(100):
        b = scipy.sparse.dok_matrix(numpy.round(numpy.random.random((2, 3))), dtype='uint8');
        c = scipy.sparse.vstack((b, b), format='dok', dtype='uint8');
    clock = time.time()-clock;
    print clock
    
def print_output_nyt_statistics():
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer();
    
    words = nltk.probability.FreqDist();
    for line in open('../data/nyt/voc.dat', 'r'):
        line = line.lower();
        line = re.sub(r'-', ' ', line);
        line = re.sub(r'[^a-z ]', '', line);
        #line = re.sub(r' +', '', line);
        
        for word in line.split():
            if word in nltk.corpus.stopwords.words('english'):
                continue;
            word = stemmer.stem(word);
            if word in nltk.corpus.stopwords.words('english'):
                print word
            if len(word)>=20 or len(word)<=2:
                continue;
            words.inc(word);

    print len(words.keys());

def reverse_cumulative_sum_matrix_over_axis(matrix, axis):
    cumulative_sum = numpy.zeros(matrix.shape);
    (k, n) = matrix.shape;
    if axis == 1:
        for j in xrange(n - 2, -1, -1):
            cumulative_sum[:, j] = cumulative_sum[:, j + 1] + matrix[:, j + 1];
    elif axis == 0:
        for i in xrange(k - 2, -1, -1):
            cumulative_sum[i, :] = cumulative_sum[i + 1, :] + matrix[i + 1, :];

    return cumulative_sum;

def measure_time():    
    a = numpy.random.random((1000000, 1));
    #print a
    
    clock = time.time();
    for i in xrange(1000):
        b = numpy.fliplr(numpy.cumsum(numpy.fliplr(a), 1));
        b[:, 0] = 0;
        b = numpy.roll(b, -1, 0);
    #print b;
    clock = time.time()-clock;
    print clock
    
    clock = time.time();
    for i in xrange(1000):
        c = reverse_cumulative_sum_matrix_over_axis(a, 1);
    #print c
    clock = time.time()-clock;
    print clock
    
    print numpy.all(b==c);
    
def testSimpleGoodTuringProbDist():
    f = nltk.probability.FreqDist([1, 2, 1, 1, 2, 3]);
    print f.Nr(1), f.Nr(2), f.Nr(3)
    p = nltk.probability.SimpleGoodTuringProbDist(f, 5);
    print p.prob(1), p.discount();
    print p.prob(2), p.discount();
    print p.prob(3), p.discount();
    print p.prob(4), p.discount();
    print p.prob(5), p.discount();

def compute_exp_weights(self):
    psi_nu_1 = scipy.special.psi(self._nu_1);
    psi_nu_2 = scipy.special.psi(self._nu_2);
    psi_nu_all = scipy.special.psi(self._nu_1 + self._nu_2);
    
    assert(psi_nu_1.shape == (self._number_of_topics, self._truncation_size-1));
    assert(psi_nu_2.shape == (self._number_of_topics, self._truncation_size-1));
    assert(psi_nu_all.shape == (self._number_of_topics, self._truncation_size-1));

    aggregate_psi_nu_2_minus_psi_nu_all = numpy.zeros((self._number_of_topics, self._truncation_size));
    assert(aggregate_psi_nu_2_minus_psi_nu_all.shape == (self._number_of_topics, self._truncation_size));
    
    for t in xrange(self._truncation_size-1):
        aggregate_psi_nu_2_minus_psi_nu_all[:, t+1] = aggregate_psi_nu_2_minus_psi_nu_all[:, t] + psi_nu_2[:, t] - psi_nu_all[:, t];

    return psi_nu_1 - psi_nu_all, psi_nu_2 - psi_nu_all, aggregate_psi_nu_2_minus_psi_nu_all
"""

if __name__ == '__main__':
    import string
    s = "string. With. Punctuation?" # Sample string 
    out = s.translate(string.maketrans("",""), string.punctuation)
    print out
    
    test();