def parse_data(documents_file, vocabulary_file=None):
    import codecs
    
    type_to_index = {};
    index_to_type = {};
    if (vocabulary_file!=None):
        input_file = codecs.open(vocabulary_file, mode='r', encoding='utf-8');
        for line in input_file:
            line = line.strip().split(0);
            assert line not in type_to_index, "duplicate type for %s" % line;
            type_to_index[line] = len(type_to_index);
            index_to_type[len(index_to_type)] = line;
        input_file.close();

    input_file = codecs.open(documents_file, mode="r", encoding="utf-8")

    doc_count = 0
    documents = {};
    
    for line in input_file:
        line = line.strip().lower();

        contents = line.split("\t");

        document = [];
        for token in contents[-1].split():
            if token not in type_to_index:
                if vocabulary_file==None:
                    type_to_index[token] = len(type_to_index);
                    index_to_type[len(index_to_type)] = token;
                else:
                    continue;
                    
            document.append(type_to_index[token]);
        
        assert len(document)>0, "document %d collapsed..." % doc_count;
        
        if len(contents)==2:
            documents[int(contents[0])] = document;
        elif len(contents)==1:
            documents[doc_count] = document;
        else:
            print ""  

        doc_count+=1
        if doc_count%10000==0:
            print "successfully import %d documents..." % doc_count;
    
    input_file.close();

    print "successfully import", len(documents), "documents..."
    return documents, type_to_index, index_to_type;

"""
this method convert a corpus from dict(list) to dict(FreqDist) format, same tokens are grouped
@return: a dict(FreqDist) data type, first indexed by the document id, then indexed by the unique tokens
@param corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
"""
def dict_list_2_dict_freqdist(documents):
    from collections import defaultdict;
    from nltk.probability import FreqDist;

    docs = defaultdict(dict)
    
    for doc in documents.keys():
        docs[doc] = FreqDist(documents[doc])
    
    return docs







#def import_de_news_data(input_file, doc_limit=-1):
#    import codecs
#    input = codecs.open(input_file, mode="r", encoding="utf-8")
#    
#    doc_count = 0
#    docs = {}
#    
#    for line in input:
#        line = line.strip().lower();
#
#        contents = line.split("\t");
#        assert(len(contents)==2);
#        docs[contents[0]] = contents[1].split()
#        
#        doc_count+=1 
#    
#        if doc_count%10000==0:
#            print "successfully import " + str(doc_count) + " documents..."
#            
#        if doc_limit > 0 and doc_count > doc_limit:
#            print("passed doc limit %i" % doc_count)
#            return docs
#    
#    print "successfully import all documents..."
#    return docs

def import_monolingual_data(input_file):
    import codecs
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    
    doc_count = 0
    docs = {}
    
    for line in input:
        line = line.strip().lower();

        contents = line.split("\t");
        assert(len(contents)==2);
        docs[int(contents[0])] = [int(item) for item in contents[1].split()];

        doc_count+=1
        if doc_count%10000==0:
            print "successfully import " + str(doc_count) + " documents..."

    print "successfully import", len(docs), "documents..."
    return docs

"""
this method convert a corpus from dict(list) to defaultdict(dict) format, similar words are grouped
@return: a defaultdict(dict) data type, first indexed by the document id, then indexed by the unique tokens
@param corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
"""
def dict_list_2_dict_dict(corpus):
    from collections import defaultdict;

    docs = defaultdict(dict)
    
    for doc in corpus.keys():
        content = {}
        for term in corpus[doc]:
            if term in content.keys():
                content[term] = content[term] + 1
            else:
                content[term] = 1
        docs[doc] = content
    
    return docs

"""
this method convert two lists to a dict, two input lists must share the same length
@return: a dict data type, keyed by the elements in list_a and valued by the elements in list_b
@param list_a: a list contains the corresponding keys
@param list_b: a list contains the corresponding values
"""
def two_lists_2_dict(list_a, list_b):
    assert len(list_a)==len(list_b)
    
    return dict(zip(list_a, list_b))
