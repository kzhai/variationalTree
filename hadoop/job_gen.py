job_template_path = "../src/vb/prior/tree/dumbo/launch.sh";

input_directory='../data/'
output_directory='../output_stream/'

#format="doc"
#dataset="fbis"
#language="zh"
#cn='sent.fbis-zh-en'

ni=50;
nm=50;
si=10;
uh="\"true\"";

count = 0;

for format in ["doc", "sent"]:
    dataset="fbis"
    for language in ['zh', 'zh-en']:
        cn = "%s.%s-%s" % (format, dataset, language);

        tn = "hownet-tree";
        
        # if language=="zh":
        #     tn='tree0'
        # elif language=="zh-en":
        #     tn="tree1"
        # else:
        #     print "new tree?"

        for nt in [10]:
            count += 1;
            
            nr=nt;
            #parameterStrings="%s-K%d-I%d-M%d-R%d" % (tn, nt, ni, nm, nr);
            
            input_stream = open(job_template_path, 'r');
            output_stream = open(cn+"-"+str(nt)+"-"+str(count)+"-alpha.sh", 'w');

            for line in input_stream:
                line = line.rstrip();
            
                if line.startswith("SET_PARAMETER"):
                    output_stream.write("CorpusName=%s\n" %cn);
                    output_stream.write("TreeName=%s\n" %tn);
                    output_stream.write("NumTopic=%d\n" %nt);
                    output_stream.write("Iterations=%d\n" %ni);
                    output_stream.write("NumMapper=%d\n" %nm);
                    output_stream.write("NumReducer=%d\n" %nr);
                    output_stream.write("SnapshotInterval=%d\n" %si);
                    output_stream.write("UpdateHyperParameter=%s\n" % uh);
                    #output_stream.write("Suffix=%s\n" %(parameterStrings));
            
                    continue;
            
                if line.startswith("SET_POST_PIPELINE"):
                    output_stream.write("$PYTHON_COMMAND -m experiments.test \\\n");
                    output_stream.write("    $ProjectDirectory/data/$CorpusName/doc.dat \\\n");
                    #output_stream.write("    $MTDevTestDirectory/%s.%s-%s/doc.dat \\\n" % ("sent", dataset, "zh"));
                    output_stream.write("    $MTOutputSubDirectory/model.docs \\\n");
                    output_stream.write("    $LocalOutputDirectory/ \\\n");
                    output_stream.write("    $LocalInputDirectory/voc.dat\n");
            
                    continue;
            
                output_stream.write(line + "\n");
            
            '''
            if format=="doc":
                output_stream.write("$PYTHON_COMMAND -m experiments.test \\");
                output_stream.write("\t$MTDevTestDirectory/%s.%s-%s/doc.dat \\" % ("sent", dataset, "zh"));
                output_stream.write("\t$MTOutputSubDirectory/model.docs \\");
                output_stream.write("\t$LocalOutputDirectory/ \\");
                output_stream.write("$LocalInputDirectory/voc.dat\n\n");
                
            elif format=="sent":
                if language=="zh-en":
                    if dataset=="fbis":
                        output_stream.write("head -n 268706 $MTOutputSubDirectory/model.docs.all > $MTOutputSubDirectory/model.docs\n");
                    elif dataset=="nist":
                        output_stream.write("head -n 1657744 $MTOutputSubDirectory/model.docs.all > $MTOutputSubDirectory/model.docs\n");
                    else:
                        print "";
                    
                elif language=="zh":
                    if format=="sent":
                        output_stream.write("$PYTHON_COMMAND -m experiments.parse_gamma $LocalOutputDirectory/gamma $MTOutputSubDirectory/model.docs\n");
                    elif format=="doc":
                        output_stream.write("$PYTHON_COMMAND -m experiments.test $MTDevTestDirectory/%s.%s-%s/doc.dat $MTOutputSubDirectory/model.docs $LocalOutputDirectory/ $LocalInputDirectory/voc.dat\n" % ("sent", dataset, language));
                    else:
                        print "unexpected format..."
    
                else:
                    print "unexpected language..."
            '''
