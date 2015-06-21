import optparse;

delimiter = '-';

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        corpus_name=None,
                        tree_name=None,
                        
                        # parameter set 2
                        number_of_iterations=-1,
                        number_of_topics=-1,

                        # parameter set 3
                        alpha=-1,
                        #default_correlation_prior=1e-2,
                        #positive_correlation_prior=1e2,
                        #negative_correlation_prior=1e-12,
                        #eta=1e-12,
                        
                        # parameter set 4
                        hybrid_mode=False,
                        update_hyperparameter=False,
                        #inference_type="cgs",
                        
                        # parameter set 5
                        snapshot_interval=10
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      help="the corpus name [None]")
    parser.add_option("--tree_name", type="string", dest="tree_name",
                      help="the tree_name file [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="total number of topics [-1]");
    parser.add_option("--number_of_iterations", type="int", dest="number_of_iterations",
                      help="total number of iterations [-1]");
                      
    # parameter set 3
    parser.add_option("--alpha", type="float", dest="alpha",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    #parser.add_option("--positive_correlation_prior", type="float", dest="positive_correlation_prior",
                      #help="positive correlation prior between words [1e2]")
    #parser.add_option("--negative_correlation_prior", type="float", dest="negative_correlation_prior",
                      #help="negative correlation prior between words [1e-12]")
    #parser.add_option("--default_correlation_prior", type="float", dest="default_correlation_prior",
                      #help="normal correlation prior between words [1e-2]")
    #parser.add_option("--eta", type="float", dest="eta",
                      #help="hyper-parameter for Dirichlet distribution of vocabulary [1e-12]")
    
    # parameter set 4
    parser.add_option("--hybrid_mode", action="store_true", dest="hybrid_mode",
                      help="enable hybrid inference mode");
    parser.add_option("--update_hyperparameter", action="store_true", dest="update_hyperparameter",
                      help="update alpha (hyper-parameter for topic Dirichlet distribution)");
    #parser.add_option("--inference_type", type="string", dest="inference_type",
                      #help="inference type [cgs] cgs-CollapsedGibbsSampling uvb-UncollapsedVariationalBayes hybrid-HybridMode");
    #parser.add_option("--inference_type", action="store_true", dest="inference_type",
    #                  help="run latent Dirichlet allocation in hybrid mode");
    
    # parameter set 5
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [vocab_prune_interval]");

    (options, args) = parser.parse_args();
    return options;