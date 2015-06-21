import numpy;

from nltk.probability import FreqDist;

#uninformed_count = 1.;
#informed_count =  100.;

#vocab = ['python', 'java', 'umiacs', 'clip', 'doctor', 'master'];
#total_vocab = xrange(6);
#total_topic = 3;
#topic_vocab = 2;

#topic = {};
#topic[0] = numpy.array([informed_count, informed_count, uninformed_count, uninformed_count, uninformed_count, uninformed_count]);
#topic[1] = numpy.array([uninformed_count, uninformed_count, informed_count, informed_count, uninformed_count, uninformed_count]);
#topic[2] = numpy.array([uninformed_count, uninformed_count, uninformed_count, uninformed_count, informed_count, informed_count]);

#for k in xrange(total_topic):
#    topic[k] = FreqDist();
#    for v in total_vocab:
#        topic[k].inc(v, uninformed_count);
#    topic[k].inc(k*topic_vocab, informed_count); 
#    topic[k].inc(k*topic_vocab+1, informed_count);
    
def generate_corpus(D=100, K=3, alpha=None):
    if alpha==None:
        alpha = numpy.random.random(K);
    #alpha = alpha/numpy.sum(alpha);

    topic = numpy.random.random((1, K));
    topic /= numpy.sum(topic);
    
    term_per_doc = 20;
    for d in xrange(D):
        doc = "";
        
        gamma = numpy.random.mtrand.dirichlet(alpha);
        for n in xrange(numpy.random.poisson(term_per_doc)):
            k = numpy.nonzero(numpy.random.multinomial(1, gamma))[0][0];
            doc += str(numpy.nonzero(numpy.random.multinomial(1, topic[k]))[0][0]) + " ";
        
        print str(d) + "\t" + doc.strip()
        
if __name__ == '__main__':
    alpha = numpy.ones(3)*0.1;
    
    generate_corpus(100, 3, alpha);