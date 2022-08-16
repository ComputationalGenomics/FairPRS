import os, argparse, time, subprocess, string
import re
import pandas as pd
import summary_stats
import prs_base
import fairPRS
from utils import results_summary

def msg(name=None):
    return '''FairPRS.py
        Compute GWAS, PCs and train/tune model
         >> python fairPRS.py -sim_data 1 --prop 10,20,70 -szs 10000,1000,400 -snps 100000 -causal 0.05 --random_seed 42 -iters 10 -tun 0 -plot_inp_dist 1 -envs 3 -model_train 0 --trait 1 --model PSD -num_pop 3 -num_pcs 8 -r 1
        Load real data, train/tune model
         >> python fairPRS.py -sim_data 0 --random_seed 42 -iters 10 -tun 1 -plot_inp_dist 0 -envs 3 -model_train 1 --trait 1 --model PSD -num_pop 3 -train_fname PRS_prsice_train_PSD_10_20_70_continuous_0.05_0.best -test_fname PRS_prsice_test_PSD_10_20_70_continuous_0.05_0.best
         *above the sample data is in the directory 'testing/' and the prefix to the PLINK format data is 'test_data'.
        '''

def parse_arguments():
    parser = argparse.ArgumentParser(usage=msg())

    parser.add_argument("-s", "--sim", dest='simulate', action='store', help="Indicate whether to use simulated data (--sim 1)."+
                        " If not using simulated data, indicate real dataset using '-d/--dir' flag.",
                        metavar="SIM")

    parser.add_argument("-d", "--dir", dest='realdata_directory', action='store', help="Put the path to the real dataset.",
                        metavar="DIR")

#     parser.add_argument("-p", "--pval", dest='pvalue', action='store', help="Enter the desired p-value threshold",
#                         metavar="PV")

    parser.add_argument("-pr", "--prop", dest='prop', action='store', help="Enter comma separated proportion of variation",
                        metavar="PROP")

    parser.add_argument("-tf", "--trait", dest='trait_flag', action='store', help="Enter the trait flag -- 1 for binary, 0 for continuous",
                        metavar="TR")

    parser.add_argument("-m", "--model", dest='model_flag', action='store', help="Enter the desired simulation model",
                        metavar="MODEL")

    parser.add_argument("-szs", "--sample_sizes", dest='sample_size_flag', action='store', help="Enter the desired number of samples for GWAS and PRS comma separated", metavar="SMPSIZE")
    
    parser.add_argument("-snps", "--num_snps", dest='num_snps_flag', action='store', help="Enter the desired number of SNPs (features) for simualtedd data", metavar="NSNPS")
    
    parser.add_argument("-envs", "--num_pop_envs", dest='envs_flag', action='store', help="Enter the number of expected of populations in data (number of training environments for IRM)", metavar="NSNPS")

    parser.add_argument("-v", "--ver", dest='verbose', action='store', help="Set for verbose output with timing profile",
                        metavar="VER")

    parser.add_argument("-pf", "--plot", dest='plot_flag', action='store', help="flag for plotting",
                        metavar="PFLAG")
    
    parser.add_argument("-rnd_sed", "--random_seed", dest='random_seed', action='store', help='Enter random seed to be used for SNP position and allele fixing.', metavar='RNDSEED')
    
    parser.add_argument("-num_pop", "--num_populations", dest='num_pop', action='store', help='Enter the number of ancestry populations to be used in the simulator.', metavar='NUMPOP')
    
    parser.add_argument("-causal", "--causal_snps_prop", dest='num_causal', action='store', help='Enter the perecentage in decimals (eg 0.1 means 10%) of the total number of SNPs to be set as causal to phenotype.', metavar='CAUSAL')
    
    parser.add_argument("-u_e", "--units_erm", dest='units_erm', action='store', help='Enter the units per linear layer of encoder/decoder (up to 5 hidden layers), comma separated ', metavar='UNITSERM')
    parser.add_argument("-u_i", "--units_irm", dest='units_irm', action='store', help='Enter the units per linear layer of encoder/decoder (up to 5 hidden layers), comma separated ', metavar='UNITSIRM')
    parser.add_argument("-lrs", "--learning_rates", dest='lrs', action='store', help='Enter the learning rate for ERM and IRM models, comma separated.', metavar='LR')
    parser.add_argument("-pen", "--penalty_multiplier", dest='pen_mult', action='store', help='Enter the penalty multiplier for IRM.', metavar='PENALTY')
    parser.add_argument("-tun", "--tuning_bool", dest='autotunning_flag', action='store', help='Enter 0 for no tuning and perform experiment the number of iterations entered, or 1 for autotuning for models.', metavar='TUNING')
    parser.add_argument("-iters", "--iterations", dest='iters', action='store', help='Enter number of iterations to run full pipeline', metavar='ITERS')
    parser.add_argument("-g", "--GWAS_PC_calc", dest='step1', action='store', help='Enter 1 if you want to compute summary stats and PCs for your data.', metavar='STEP1', default=0)
    parser.add_argument("-p", "--PRS_calc", dest='step2', action='store', help='Enter 1 if you want to compute PRS for your data.', metavar='STEP2', default=0)
    parser.add_argument("-fair", "--fairPRS_calc", dest='step3', action='store', help='Enter 1 if you want to compute FairPRS for your data.', metavar='STEP3', default=0)
    parser.add_argument("-prs_path", "--PRS_load_path", dest='PRS_load_path', action='store', help='Enter the path to your PRS files', metavar='PRSPATH')
    parser.add_argument("-model_train", "--model_training", dest='model_training', action='store', help='Do you want to trainany models? 1-yes 0-no', metavar='MODELTRAIN')
    parser.add_argument("-plot_inp_dist", "--plot_input_distributions", dest='plot_input_dists', action='store', help='Do you want to plot the PRS distributions of input data? 1- yes  0- no', metavar='PLOTINPUT', default= 1)
    parser.add_argument("-plot_out_dist", "--plot_output_distributions", dest='plot_output_dists', action='store', help='Do you want to plot the PRS and Phenotype distributions of output data? 1- yes  0- no', metavar='PLOTOUTPUT', default= 1)
    parser.add_argument("-man_plot", "--manhattan_plot", dest='man_plot', action='store', help='Do you want to plot the Manhattan plot? 1- yes  0- no', metavar='PLOTOUTPUT', default=0)
    parser.add_argument("-train_fname", "--train_data_file_name", dest='train_fname', action='store', help='Enter training PRS file name (must be stored in data/model/fname', metavar='TRAINFNAME')
    parser.add_argument("-test_fname", "--test_data_file_name", dest='test_fname', action='store', help='Enter test PRS file name (must be stored in data/model/fname)', metavar='TESTFNAME')
    parser.add_argument("-fi", "--fname_prefix_root", dest='fname_prefix_root', action='store', help='Enter prefix name for outfiles', metavar='FNAMEROOT')
    parser.add_argument("-fo", "--fname_out_root", dest='fname_out_root', action='store', help='Enter prefix name for inputfiles', metavar='FNAMEROOT')
    parser.add_argument("-gpu", "--gpu_bol", dest='gpu', action='store', help='Enter 1 for GPU or 0 for cpu (default cpu)', metavar='GPUBOOL', default=0)
    parser.add_argument("-num_pcs", "--num_pcs", dest='num_pcs', action='store', help='Enter the number of PCs', metavar='NUMPCS', default=10)
    parser.add_argument("-num_covs", "--num_covs", dest='num_covs', action='store', help='Enter the number of covariates (non-pcs)', metavar='NUMCOVS', default=3)
    parser.add_argument("-r", "--risk", dest='risk_flag', action='store', help="Set Risk flag if you want risk for one population to be greater than the others",metavar="RISK")
    parser.add_argument("-pheno_fname", "--pheno_fname", dest='pheno_path', action='store', help='Enter path for phenotype file for PRS', metavar='PHENOFNAME')
    parser.add_argument("-gwas_fname", "--gwas_fname", dest='gwas_path', action='store', help='Enter path for summary statistics file for PRS', metavar='GWASFNAME')
    parser.add_argument("-geno_fname", "--geno_fname", dest='geno_path', action='store', help='Enter path for genotype file for PRS', metavar='GENOFNAME')
    parser.add_argument("-covs_fname", "--covs_fname", dest='covs_path', action='store', help='Enter path for covariates file for PRS', metavar='COVSFNAME')
    parser.add_argument("-a", "--access", dest='access', action='store', help='Enter the access points of the pipeline to run. i.e type geno,sumstat,prs (comma separated)', metavar='ACCESS')
    parser.add_argument("-sav", "--saveres", dest='saveres', action='store', help='Enter 1 for saving results or 0 for only print out ', metavar='SAVERES', default=1)
    
#     parser.add_argument("-src_path", "--scripts_path", dest='scripts_path', action='store', help="Enter the data_sim ", metavar="DFLAG")

    args = parser.parse_args()

    return args 

if __name__ == '__main__':
    begin_time = time.time()
    args = parse_arguments()
    
    ### Prepare environment ###
    if not os.path.isdir('../data'):
        os.system('mkdir ../data')
    if not os.path.isdir('../results'):
        os.system('mkdir ../results')
    
    # Parse to local variables
    iters = int(args.iters)

    if int(args.trait_flag) == 1:
        pheno = 'continuous'
        binary_target = 'F'
        trait_flag = 1
    elif int(args.trait_flag) == 0:
        pheno = 'Binary'
        binary_target = 'T'
        trait_flag = 0

    load_data = False
    if args.PRS_load_path:
        load_data = True
        args.step3 = 1

    if int(args.step3)==1:
        load_data = True
    
    if args.prop is not None:
        prop_genetic,prop2_environmental,prop3_noise = args.prop.split(',',2)
    if args.sample_size_flag is not None:
        train_size,PRS_train_size,PRS_test_size = args.sample_size_flag.split(',',2)

    if args.fname_prefix_root is not None:
        fname_root = args.fname_prefix_root
    
    if args.fname_out_root is not None:
        fname_root_out = args.fname_out_root
        if args.fname_prefix_root is None:
            fname_root = fname_root_out

    model_flag = args.model_flag
    num_pcs = int(args.num_pcs)
    num_envs = int(args.envs_flag)
    num_covs = int(args.num_covs)
    man_plot = bool(int(args.man_plot))
    gwas_path = args.geno_path
    pheno_path = args.pheno_path
    geno_path = args.geno_path
    covs_path = args.covs_path
    access = args.access.split(',')
    gpu = bool(int(args.gpu))
    rnd_state = int(args.random_seed)
    plot_input_dists = bool(int(args.plot_input_dists))
    plot_output_dists = bool(int(args.plot_output_dists))
    saveres = bool(int(args.saveres))




    for itr in range(iters):
        if ('geno' in access):
            summary_stats.summary_stats(model_flag, fname_root, num_pcs, itr, man_plot)
        if ('sumstats' in access):
            prs_base.prs_base(fname_root, model_flag, gwas_path, geno_path, pheno_path, covs_path, binary_target, itr)
        if ('prs' in access):
            if args.realdata_directory is not None:
                real_data_df = pd.read_csv(args.realdata_directory)
            else: 
                real_data_df = None
                fname_root_out = fname_root
            results_dict_erm,results_dict_irm = fairPRS.fairprs(real_data_df, fname_root, itr = itr,
                        plot_input_dists = plot_input_dists, plot_output_dists = plot_output_dists, test_size = 0.10 , val_size = 0.22,
                        rnd_state = rnd_state, gpu = gpu, num_pcs = num_pcs, num_envs = num_envs, model_flag = model_flag,
                        trait_flag = trait_flag, num_covs = num_covs, fname_root_out = fname_root_out)
            
            # Results summary and return full results dictionary 
            results_df_erm = results_summary(results_dict_erm, 'ERM', model_flag, fname_root_out, itr, saveres)
            results_df_irm = results_summary(results_dict_irm, 'IRM', model_flag, fname_root_out, itr, saveres)
                
            if (itr+1)%10 == 0:
                print(f'{itr} iterations completed.')
                print('='*40)
                
    print("--- Total run time in %s seconds ---" % (time.time() - begin_time))