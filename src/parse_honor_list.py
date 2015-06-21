import numpy;
import scipy;
import random;
import itertools;
import nltk;
import time;

def add_title():
    input = '../data/ap/doc.dat';
    output = open('../data/ap/doc-title.dat', 'w');
    index = 1;
    for line in open(input, 'r'):
        line = line.strip();
        
        content = line.split();
        new_content = [];
        for word in content:
            if word in nltk.corpus.stopwords.words('english'):
                continue;
            else:
                new_content.append(word);
        output.write(str(index) + "\t" + " ".join(new_content) + "\n");
        
        index += 1;
        
def filter_unknows():
    input = '../data/tau-map-full-finite-ignore-trans.txt';
    output = open('../data/tau-map-full-finite-ignore-trans.dat', 'w');
    index = 1;
    for line in open(input, 'r'):
        line = line.strip();
        
        if line.startswith('UNK'):
            continue;
        else:
            output.write(line + "\n");
        
def parse_honor_list():

    #vocabulary_file = '../data/ap/voc.dat'
    #honor_list_file = '../data/ap/honor_english.txt';
    #inform_prior_file = '../data/ap/category.dat'
    
    vocabulary_file = '/fs/clip-lsbi/Mr.LDA/blog/voc.dat'
    honor_list_file = '/fs/clip-lsbi/Mr.LDA/blog/honor_english.txt';
    inform_prior_file = '/fs/clip-lsbi/Mr.LDA/blog/category.dat'
    
    informed_prior = {};
    honor_list = {}
    title_block_flag = False;
    for line in open(honor_list_file, 'r'):
        line = line.strip();
        
        if line.startswith('%'):
            title_block_flag = not title_block_flag;
            continue;
        
        '''
        content = line.split();
        if title_block_flag:
            assert(len(content)==2);
            honor_list[content[0].strip()] = [];
        else:
            #print line
            #print word, content[1:];
            word = content[0].strip();
            for word in content[1:]:
                honor_list[word.strip()].append(word);
        '''
        
        content = line.split();
        if title_block_flag:
            assert(len(content)==2);
            informed_prior[content[0].strip()] = [];
        else:
            word = content[0].strip();
            honor_list[word] = content[1:];
                
    #for word in honor_list.keys():
    #    print word, honor_list[word]
    
    for line in open(vocabulary_file):
        line = line.strip();
        line = line.split()[1];
        for word in honor_list.keys():
            if '*' in word and line.startswith(word.strip('*')):
                for category in honor_list[word]:
                    informed_prior[category.strip()].append(line);
            if '*' not in word and line==word:
                for category in honor_list[word]:
                    informed_prior[category.strip()].append(line);
    
    output = open(inform_prior_file, 'w');
    for word in informed_prior.keys():
        output.write(word + "\t" + (" ".join(informed_prior[word])) + "\n");
        #print informed_prior[word]
        #print word, informed_prior[word]

def parse_topic_word_list():

    #vocabulary_file = '../data/ap/voc.dat'
    #honor_list_file = '../data/ap/honor_english.txt';
    
    #honor_list = {}
    title_block_flag = False;

    #topic_word_list = '../data/ap/mrlda-ap-K25I40-500.beta';
    
    inform_prior_file = '../data/ap/category2.dat'
    informed_prior = {};
    for line in open(inform_prior_file, 'r'):
        line = line.strip();
        
        content = line.split("\t");
        category = content[0].strip();
        
        for word in content[1].split():
            word = word.strip()
            if word not in informed_prior.keys():
                informed_prior[word] = []
            informed_prior[word].append(category);
            
    #for key in informed_prior:
    #    print key, informed_prior[key];
        
    topic_id = -1;
    category_list = nltk.probability.FreqDist();
    topic_word_list = '../data/ap/mrlda-ap-K25I40-500.beta';
    for line in open(topic_word_list, 'r'):
        line = line.strip();
        if line.startswith('=========='):
            continue;
        elif line.startswith('Top ranked'):
            if topic_id>=0:
                print "Topic", topic_id
                category_list._sort_keys_by_value();
                print category_list
            topic_id = int(line.split()[-1]);
            category_list = nltk.probability.FreqDist();
            continue;
        else:
            content = line.split();
            value = numpy.exp(float(content[1]));
            type = content[0];
            if type in informed_prior.keys():
                for category in informed_prior[type]:
                    #category_list.inc(category, value);
                    category_list.inc(category, 1);

def retrieve_category_information():

    #vocabulary_file = '../data/ap/voc.dat'
    #honor_list_file = '../data/ap/honor_english.txt';
    
    #honor_list = {}
    title_block_flag = False;

    topic_word_list = '../data/ap/mrlda-ap-K25I40-500.beta';
    
    inform_prior_file = '../data/ap/category2.dat'
    informed_prior = {};
    i=1;
    for line in open(inform_prior_file, 'r'):
        line = line.strip();
        
        content = line.split("\t");
        category = content[0].strip();
        
        informed_prior[i] = category;
        i += 1;
        
    topic_word_list_original = '../data/ap/mrlda-ap-K25I40-500.beta';
    topic_word_list_retrieved = '../data/ap/mrlda-ap-K25I40-500.beta.retrieved';
    output = open(topic_word_list_retrieved, 'w');
    direct_to_output = True;
    for line in open(topic_word_list_original, 'r'):
        line = line.strip();
        if line.startswith('Top ranked'):
            tokens = line.split();
            topic_id = int(tokens[-1]);
            if topic_id in informed_prior.keys():
                tokens[-1] = informed_prior[topic_id];
                line = " ".join(tokens);
                direct_to_output = True;
            else:
                direct_to_output = False;
        if direct_to_output:
            output.write(line+"\n");


def highlight_category_information():
    inform_prior_file = '../data/blog/category2.dat'
    informed_prior = {};
    for line in open(inform_prior_file, 'r'):
        line = line.strip();
        
        content = line.split("\t");
        category = content[0].strip();
        
        informed_prior[category] = content[1].split();
  
    print informed_prior
            
    topic_word_list_original = '../data/blog/blog_informed_prior_50.map';
    topic_word_list_retrieved = '../data/blog/blog_informed_prior_50.highlighted';
    output = open(topic_word_list_retrieved, 'w');
    for line in open(topic_word_list_original, 'r'):
        line = line.strip();
        if line.startswith('Top ranked'):
            tokens = line.split();
            topic_id = tokens[-1];
            #if topic_id in informed_prior.keys():
            #    line = " ".join(tokens);
        elif line.startswith('=========='):
            output.write(line + "\n");
            continue;
        else:
            tokens = line.split();
            if topic_id in informed_prior.keys() and tokens[0] in informed_prior[topic_id]:
                line = '*' + line;
        output.write(line+"\n");

def parse_blog_data():
    category_file = open('../data/blog/honor_english.txt', 'r');
    category = {};
    skip = True;
    for line in category_file:
        if line.startswith("%"):
            skip = False;
        else:
            content = line.split();
            category[content[0].strip()] = content[1].strip();
        if skip:
            break;
    
    index = {}
    input = open('../data/blog/category.dat', 'r');
    output = open('../data/blog/category2.dat', 'w');
    for line in input:
        content = line.split();
        content[0] = category[content[0]];
        index[len(index)+1]=content[0];
        output.write(content[0] + "\t" + " ".join(content[1:]) + "\n");
        
    input = open('../data/blog/blog_informed_prior_50.txt', 'r');
    output = open('../data/blog/blog_informed_prior_50.map', 'w');
    for line in input:
        if line.startswith("Top ranked"):
            content = line.split();
            if int(content[-1]) <= len(index): 
                content[-1] = index[int(content[-1])];
            output.write(" ".join(content) + "\n");
        else:
            output.write(line);

if __name__ == '__main__':
    #parse_honor_list();
    parse_blog_data();
    highlight_category_information();