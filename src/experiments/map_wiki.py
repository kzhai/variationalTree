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
    map_input_file_path = sys.argv[1];
    zh_input_file_path = sys.argv[2];
    en_input_file_path = sys.argv[3];

    map_output_file_path = sys.argv[4];
    zh_output_file_path = sys.argv[5];
    en_output_file_path = sys.argv[6];

    candidate_zh_ids = set();
    candidate_en_ids = set();
    id_mapping = set();
    map_input_stream = codecs.open(map_input_file_path, 'r', 'utf-8');
    mapping_counter = 0;
    for line in map_input_stream:
        mapping_counter += 1;
        line = line.strip();
        tokens = line.split("\t");
        assert len(tokens)==2;

        zh_ids = set(tokens[0].split());
        en_ids = set(tokens[1].split());

        candidate_zh_ids = candidate_zh_ids | zh_ids;
        candidate_en_ids = candidate_en_ids | en_ids;

        id_mapping.add((" ".join(zh_ids), " ".join(en_ids)))
        if mapping_counter % 10000==0:
            print "successfully import %d mappings" % (mapping_counter);
            
    print "successfully import %d doc mappings with %d zh-docs and %d en-docs" % (len(id_mapping), len(candidate_zh_ids), len(candidate_en_ids));
    
    zh_id_doc_dict = {};
    zh_input_stream = codecs.open(zh_input_file_path, 'r', 'utf-8');
    zh_id_doc_counter = 0;
    for line in zh_input_stream:
        zh_id_doc_counter += 1;
        line = line.strip();
        tokens = line.split("\t");
        if len(tokens)!=2:
            continue

        title_tokens = tokens[0].split();
        title = title_tokens[0];
        if not title.isdigit():
            continue;
        if title not in candidate_zh_ids:
            continue;

        content = " ".join(title_tokens[2:]) + " " + tokens[1];
        zh_id_doc_dict[title] = content;

        if zh_id_doc_counter % 10000==0:
            print "successfully import %d zh docs" % (zh_id_doc_counter);
    print "successfully import %d zh-doc" % (len(zh_id_doc_dict));

    en_id_doc_dict = {};
    en_input_stream = codecs.open(en_input_file_path, 'r', 'utf-8');
    en_id_doc_counter = 0;
    for line in en_input_stream:
        en_id_doc_counter += 1;
        line = line.strip();
        tokens = line.split("\t");
        if len(tokens)!=2:
            continue

        title_tokens = tokens[0].split();
        title = title_tokens[0];
        if not title.isdigit():
            continue;
        if title not in candidate_en_ids:
            continue;

        content = " ".join(title_tokens[2:]) + " " + tokens[1];
        en_id_doc_dict[title] = content;

        if en_id_doc_counter % 10000==0:
            print "successfully import %d en docs" % (en_id_doc_counter);
    print "successfully import %d en-doc" % (len(en_id_doc_dict));

    map_output_stream = codecs.open(map_output_file_path, 'w', 'utf-8');
    zh_output_stream = codecs.open(zh_output_file_path, 'w', 'utf-8');
    en_output_stream = codecs.open(en_output_file_path, 'w', 'utf-8');
    mapping_counter = 0;
    for (zh_ids, en_ids) in id_mapping:
        mapping_counter += 1;
        zh_doc = "";
        for zh_id in zh_ids.split():
            if zh_id not in zh_id_doc_dict:
                continue;
            zh_doc += zh_id_doc_dict[zh_id] + " ";
        zh_doc = re.sub(white_space_regex, " ", zh_doc);
        zh_doc = zh_doc.strip()


        en_doc = "";
        for en_id in en_ids.split():
            if en_id not in en_id_doc_dict:
                continue;
            en_doc += en_id_doc_dict[en_id] + " ";
        en_doc = re.sub(white_space_regex, " ", en_doc);
        en_doc = en_doc.strip()

        if len(zh_doc)==0 or len(en_doc)==0:
            continue;
        zh_output_stream.write("%s\n" % (zh_doc));
        en_output_stream.write("%s\n" % (en_doc));
        
        map_output_stream.write("%s\t%s\n" % (zh_ids, en_ids));

        if mapping_counter % 10000==0:
            print "successfully export %d mappings" % (mapping_counter);

if __name__ == '__main__':
    main()
