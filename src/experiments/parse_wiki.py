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

minimum_document_token_length = 10;

ordered_pattern_dictionary = collections.OrderedDict();

ordered_pattern_dictionary[re.compile(r"http(s)?://[\S]+")] = " ";

ordered_pattern_dictionary[re.compile(r"&#39;")] = "'";
ordered_pattern_dictionary[re.compile(r"&lt;")] = "<";
ordered_pattern_dictionary[re.compile(r"&gt;")] = ">";
ordered_pattern_dictionary[re.compile(r"&quot;")] = "\"";
ordered_pattern_dictionary[re.compile(r"&nbsp;")] = " ";

ordered_pattern_dictionary[re.compile(r"<math>(?P<math>.*?)</math>")] = " ";
ordered_pattern_dictionary[re.compile(r"<sup>(?P<sup>.*?)</sup>")] = " ";
ordered_pattern_dictionary[re.compile(r"<sub>(?P<sub>.*?)</sub>")] = " ";
ordered_pattern_dictionary[re.compile(r"<blockquote>(?P<blockquote>.*?)</blockquote>")] = " ";

ordered_pattern_dictionary[re.compile(r"<ref name(?P<name>.*?)/>")] = " ";
ordered_pattern_dictionary[re.compile(r"<ref name(?P<name>.*?)>(?P<reference>.*?)</ref>")] = " ";
ordered_pattern_dictionary[re.compile(r"<ref>(?P<reference>.*?)</ref>")] = " ";
ordered_pattern_dictionary[re.compile(r"<ref(?P<reference>.*?)>")] = " ";

ordered_pattern_dictionary[re.compile(r"<div (?P<class>.*?)/>")] = " ";
ordered_pattern_dictionary[re.compile(r"<div (?P<class>.*?)>(?P<div>.*?)</div>")] = " ";
ordered_pattern_dictionary[re.compile(r"<div>(?P<div>.*?)</div>")] = " ";
ordered_pattern_dictionary[re.compile(r"<div(?P<div>.*?)>")] = " ";

ordered_pattern_dictionary[re.compile(r"<noinclude>(?P<noinclude>.*?)</noinclude>")] = " ";

ordered_pattern_dictionary[re.compile(r"<!--(?P<comment>.*?)-->")] = " ";

ordered_pattern_dictionary[re.compile(r"<li>(?P<list>.*?)</li>")] = " ";
ordered_pattern_dictionary[re.compile(r"<ul>")] = " ";
ordered_pattern_dictionary[re.compile(r"</ul>")] = " ";

ordered_pattern_dictionary[re.compile(r"\{\{(?P<redirect>.*?)\}\}")] = " ";
ordered_pattern_dictionary[re.compile(r"\{\|(?P<meta>.*?)\|\}")] = " ";

ordered_pattern_dictionary[re.compile(r"\[\[[f|F]ile:(?P<underline>.*?)\]\]")] = " ";
ordered_pattern_dictionary[re.compile(r"\[\[(?P<underline>.*?)\]\]")] = " \\1 ";

#alpha_regex = re.compile(r'\A[a-zA-Z\s]*\Z');
ordered_pattern_dictionary[re.compile(r'[\s]+')] = " ";

def main():
    input_file_path = sys.argv[1];
    output_file_path = sys.argv[2];
    
    input_stream = codecs.open(input_file_path, 'r', 'utf-8');
    output_stream = codecs.open(output_file_path, 'w', 'utf-8');
    
    title_section = False;
    content = [];
    title = [];
    for line in input_stream:
        transcript = line.strip();
        if transcript=="----------------------------------------":
            title_section = (not title_section);
            
            if title_section:
                title_text = " ".join(title);
                content_text = " ".join(content);
                
                for regex_pattern in ordered_pattern_dictionary:
                    try:
                        content_text = re.sub(regex_pattern, ordered_pattern_dictionary[regex_pattern], content_text);
                    except sre_constants.error:
                        print regex_pattern.pattern
                        sys.exit();
                content_text = content_text.strip();
                
                if len(content_text.split())>=minimum_document_token_length:
                    output_stream.write("%s\t%s\n" % (title_text, content_text));
                    
                content = [];
                title = [];
                
            continue;
            
        if title_section:
            title.append(transcript);
        else:
            transcript = transcript.strip();
            if len(transcript)>0:
                content += transcript.split();
                
    title_text = " ".join(title);
    content_text = " ".join(content);
    
    for regex_pattern in ordered_pattern_dictionary:
        try:
            content_text = re.sub(regex_pattern, ordered_pattern_dictionary[regex_pattern], content_text);
        except sre_constants.error:
            print regex_pattern.pattern
            sys.exit();
    content_text = content_text.strip();
    
    if len(content_text.split())>=minimum_document_token_length:
        output_stream.write("%s\t%s\n" % (title_text, content_text));
    
if __name__ == '__main__':
    main()