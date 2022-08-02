import os, argparse, time, subprocess, string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from qmplot import manhattanplot
from functools import reduce
from scipy.special import rel_entr
from statsmodels.stats.weightstats import ztest
import warnings
import pickle
import statsmodels.api as sm
import hyperparam_tuning_pcs as ht
import legacy_train_test as leg_methods

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
    parser.add_argument("-g", "--GWAS_PC_calc", dest='step1', action='store', help='Enter 1 if you want to compute sunmary stats and PCs for your data.', metavar='STEP1', default=0)
    parser.add_argument("-prs_path", "--PRS_load_path", dest='PRS_load_path', action='store', help='Enter the path to your PRS files', metavar='PRSPATH')
    parser.add_argument("-model_train", "--model_training", dest='model_training', action='store', help='Do you want to trainany models? 1-yes 0-no', metavar='MODELTRAIN')
    parser.add_argument("-plot_inp_dist", "--plot_input_distributions", dest='plot_input_dists', action='store', help='Do you want to plot the PRS distributions of input data? 1- yes  0- no', metavar='PLOTINPUT')
    parser.add_argument("-train_fname", "--train_data_file_name", dest='train_fname', action='store', help='Enter training PRS file name (must be stored in data/model/fname', metavar='TRAINFNAME')
    parser.add_argument("-test_fname", "--test_data_file_name", dest='test_fname', action='store', help='Enter test PRS file name (must be stored in data/model/fname)', metavar='TESTFNAME')
    parser.add_argument("-fo", "--fname_root", dest='fname_root', action='store', help='Enter prefix name for outfiles', metavar='FNAMEROOT')
    parser.add_argument("-gpu", "--gpu_bol", dest='gpu', action='store', help='Enter 1 for GPU or 0 for cpu (default cpu)', metavar='GPUBOOL', default=0)
    parser.add_argument("-num_pcs", "--num_pcs", dest='num_pcs', action='store', help='Enter the number of PCs', metavar='NUMPCS', default=10)
    parser.add_argument("-r", "--risk", dest='risk_flag', action='store', help="Set Risk flag if you want risk for one population to be greater than the others",metavar="RISK")
    
#     parser.add_argument("-src_path", "--scripts_path", dest='scripts_path', action='store', help="Enter the data_sim ", metavar="DFLAG")

    args = parser.parse_args()

    return args

# Plotting PRS distributions by pop
def plot_prs_dist(prs_fname = None ,popids_fname= None, PRS_df = None, popids_df = None, fname_mod = None, path = 'data/', mod_name=None, simrel=None):
    sns.set_style('darkgrid')
    if popids_df is None:
        pop_ids = pd.read_csv(str(path+'/'+popids_fname+'_pop_ids_.txt'),header = None)
    else:
        pop_ids = popids_df
    if PRS_df is None:
        PRS = pd.read_csv(str(path+'/'+prs_fname),sep = '\s+')
    else:
        PRS = PRS_df
    PRS['pop_id'] = pop_ids.values
    if simrel == 1:
        PRS['pop_id'] = PRS['pop_id'].map(dict(zip(range(1,27),string.ascii_lowercase)))
        sns.kdeplot(x=PRS.PRS,hue = PRS.pop_id)
    else:
        color_palette = {
            'White': 'slategrey',
            'Mixed': 'orange',
            'Asian': 'forestgreen',
            'Black': 'saddlebrown',
            'Not Known': 'khaki',
            'Mxd': 'lightsalmon'
        }
        sns.kdeplot(x=PRS.PRS,hue = PRS.pop_id, palette = color_palette)
    if fname_mod is None:
        plt.savefig(str('results/'+mod_name+'/plots/'+popids_fname+'.png'))
    else:
        plt.savefig(str('results/'+mod_name+'/plots/'+fname_mod+'.png'))
    plt.clf()
def plot_pheno_dist(prs_fname = None ,popids_fname= None, PRS_df = None, popids_df = None, fname_mod = None, path = 'data/', mod_name=None, simrel=None):
    sns.set_style('darkgrid')
    if popids_df is None:
        pop_ids = pd.read_csv(str(path+'/'+popids_fname+'_pop_ids_.txt'),header = None)
    else:
        pop_ids = popids_df
    if PRS_df is None:
        PRS = pd.read_csv(str(path+'/'+prs_fname),sep = '\s+')
    else:
        PRS = PRS_df
    PRS['pop_id'] = pop_ids.values
    if simrel ==1:
        PRS['pop_id'] = PRS['pop_id'].map(dict(zip(range(1,27),string.ascii_lowercase)))
        sns.kdeplot(x=PRS.Phenotype,hue = PRS.pop_id)
    else:
        color_palette = {
            'White': 'slategrey',
            'Mixed': 'orange',
            'Asian': 'forestgreen',
            'Black': 'saddlebrown',
            'Not Known': 'khaki',
            'Mxd': 'lightsalmon'
        }
        sns.kdeplot(x=PRS.Phenotype,hue = PRS.pop_id, palette = color_palette)
    if fname_mod is None:
        plt.savefig(str('results/'+mod_name+'/plots/'+popids_fname+'.png'))
    else:
        plt.savefig(str('results/'+mod_name+'/plots/'+fname_mod+'.png'))
    plt.clf()

# Dataset class for pytorch
class PRS(Dataset):
    """
    Load PRS scores and phenotypes
    """
    def __init__(self, df_data_x,df_data_y,df_data_p,df_data_a):
        # load data 
        self.prs = torch.from_numpy(df_data_x).float()
        self.pheno = torch.from_numpy(df_data_y).float()
        self.ancestry = torch.from_numpy(df_data_a).int()
        self.pcs = torch.from_numpy(df_data_p).float()

    def __len__(self):
        return len(self.prs)

    def __getitem__(self, index):
        return self.prs[index], self.pheno[index], self.pcs[index], self.ancestry[index]


if __name__ == '__main__':
    begin_time = time.time()
    args = parse_arguments()
    
    ### Prepare environment ###
    if not os.path.isdir('./data'):
        os.system('mkdir data')
    if not os.path.isdir('./results'):
        os.system('mkdir results')
    
    # Parse to local variables
    iters = int(args.iters)

    if int(args.trait_flag) == 1:
        pheno = 'continuous'
        binary_target = 'F'
    elif int(args.trait_flag) == 0:
        pheno = 'Binary'
        binary_target = 'T'

    load_data = False
    if args.PRS_load_path:
        load_data = True
        args.model_training = 1

    if int(args.model_training)==1:
        load_data = True
    
    if args.prop is not None:
        prop_genetic,prop2_environmental,prop3_noise = args.prop.split(',',2)
    if args.sample_size_flag is not None:
        train_size,PRS_train_size,PRS_test_size = args.sample_size_flag.split(',',2)

    if args.fname_root is not None:
        fname_root = args.fname_root
    
    # Results dictionary   
    results_erm_tun = {}
    results_irm_tun = {}

    # Create directory for corresponding model if it doesn't exist
    if not os.path.isdir(str('./data/'+args.model_flag)):
        os.system(str('mkdir data/'+args.model_flag))
            
    for itr in range(iters):
        if bool(int(args.step1)):
            # Set up common name root for all files
            if args.prefix is not None:
                fname_root = args.prefix
            else:
                fname_root = args.geno_load_path.split('/')[-1]

            ### Compute GWAS ###
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Computing GWAS <<<<<<<<<<<<<<<<<<<<<<')
            # Compute PCS / covariates
            tera_pca_train_cmd = '~/TeraPCA/TeraPCA.exe -bfile data/'+args.model_flag+'/'+fname_root+'_train_'+str(itr)+' -nsv '+args.num_pcs+' -filewrite 1 -prefix data/'+args.model_flag+'/'+fname_root+'_train_PCA_out_'+str(itr)
            tera_pca_prs_train_cmd = '~/TeraPCA/TeraPCA.exe -bfile data/'+args.model_flag+'/'+fname_root+'_PRS_train_'+str(itr)+' -nsv '+args.num_pcs+' -filewrite 1 -prefix data/'+args.model_flag+'/'+fname_root+'_PRS_train_PCA_out_'+str(itr)
            tera_pca_prs_test_cmd = '~/TeraPCA/TeraPCA.exe -bfile data/'+args.model_flag+'/'+fname_root+'_PRS_test_'+str(itr)+' -nsv '+args.num_pcs+' -filewrite 1 -prefix data/'+args.model_flag+'/'+fname_root+'_PRS_test_PCA_out_'+str(itr)
            subprocess.call(tera_pca_train_cmd, shell=True)
            subprocess.call(tera_pca_prs_train_cmd, shell=True)
            subprocess.call(tera_pca_prs_test_cmd, shell=True)
            #Load covariates and adapt to plink format - covariate
            cov = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_train_PCA_out_'+str(itr)+'_singularVectors.txt'), sep='\t')
            cov.insert(1, 'IID', cov['FID'])
            cov_np = cov.to_numpy(dtype='str')
            cov_out = np.insert(cov_np,0,cov.columns.values, axis =0)
            np.savetxt(str('data/'+args.model_flag+'/'+fname_root+'_PCA_out_covariates_'+str(itr)+'.cov'), cov_out, delimiter='\t', fmt='%s')

            # Run GWAS with plink GLM
            gwas_cmd = 'plink2 --bfile data/'+args.model_flag+'/'+fname_root+'_train_'+str(itr)+' --glm --out data/'+args.model_flag+'/'+fname_root+'_GLM_results_'+str(itr)+' --covar data/'+args.model_flag+'/'+fname_root+'_PCA_out_covariates_'+str(itr)+'.cov'
            subprocess.call(gwas_cmd, shell=True)

            # Update GLM (GWAS) headers for standard use on PRS software and Manhattan plotting library
            nam = 'data/'+args.model_flag+'/'+fname_root+'_GLM_results_'+str(itr)+'.PHENO1.glm.linear'
            GLM = pd.read_csv(nam, sep = '\s+')
            GLM.columns = ['CHR', 'BP', 'SNP', 'A2', 'ALT','A1', 'TEST','OBS_CT', 'BETA', 'SE', 'T_STAT', 'P']
            # Drop all duplicated SNPs / only leave values where TEST column = ADD (Band aid fix)
            GLM = GLM.loc[GLM['TEST'] =='ADD']
            # Filling Na values with column means
            GLM['BETA'] = GLM['BETA'].fillna(GLM['BETA'].mean())
            GLM['SE'] = GLM['SE'].fillna(GLM['SE'].mean())
            GLM['T_STAT'] = GLM['T_STAT'].fillna(GLM['T_STAT'].mean())
            GLM['P'] = GLM['P'].fillna(GLM['P'].mean())
            GLM.to_csv(str('data/'+args.model_flag+'/'+fname_root+'_GLM_DF_'+str(itr)+'.txt'), sep='\t', index = False)

            if bool(int(args.man_plot)):
            # Plot GWAS - Manhattan plot 
                print('>>>>>>>>>>>>>>>>>> Plotting Manhattan plot for GWAS <<<<<<<<<<<<<<<<<<<<<<')

                ## Need to find alternative way to qmplot, takes too long to ouput image !!!
                nam_k = str(prop_genetic)+'_'+str(prop2_environmental)+'_'+str(prop3_noise)+'_'+str(pheno)
                man_plot_cmd = 'qmplot -I '+nam+' -T '+fname_root+' --dpi 300 -O '+'results/'+fname_root
                subprocess.call(man_plot_cmd, shell=True)
                print('>>>> Saved on Results folder <<<<<')


        if bool(int(args.step2)):
            ### Compute PRS ###
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Computing PRS <<<<<<<<<<<<<<<<<<<<<<')

            # Calculate train and test PRS
            ## Intermediate file (phenotype) for PRS calculation
            if args.pheno_path is None:
                ### Train
                pheno_out = pd.read_csv('data/'+args.model_flag+'/'+fname_root+'_PRS_train_'+str(itr)+'.fam', sep = '\s+', header = None)
                pheno_out = pheno_out.iloc[:,[0,1,5]]
                pheno_out.columns  = ['FID', 'IID', 'PHENO']
                pheno_out.to_csv(str('data/'+args.model_flag+'/'+fname_root+'_pheno_out_train_'+str(itr)+'.txt'), sep='\t', index = False)
                ### Test
                pheno_out = pd.read_csv('data/'+args.model_flag+'/'+fname_root+'_PRS_test_'+str(itr)+'.fam', sep = '\s+', header = None)
                pheno_out = pheno_out.iloc[:,[0,1,5]]
                pheno_out.columns  = ['FID', 'IID', 'PHENO']
                pheno_out.to_csv(str('data/'+args.model_flag+'/'+fname_root+'_pheno_out_test_'+str(itr)+'.txt'), sep='\t', index = False)


            ## PRSice commands - Note: change PRSice location to relaive path!
            if args.gwas_path is None:
                base_data_path = 'data/'+args.model_flag+'/'+fname_root+'_GLM_DF_'+str(itr)+'.txt'
            else:
                base_data_path = args.gwas_path
            
            if args.geno_path is None:
                target_train_data_path = 'data/'+args.model_flag+'/'+fname_root+'_PRS_train_'+str(itr)
                target_test_data_path = 'data/'+args.model_flag+'/'+fname_root+'_PRS_test_'+str(itr)
            else:
                target_train_data_path = args.target_path.split(' ')[0]   
                target_test_data_path = args.target_path.split(' ')[1]      
            
            if args.pheno_path is None:
                pheno_train_data_path = 'data/'+args.model_flag+'/'+fname_root+'_pheno_out_train_'+str(itr)+'.txt'
                pheno_test_data_path = 'data/'+args.model_flag+'/'+fname_root+'_pheno_out_test_'+str(itr)+'.txt'
            else:
                pheno_train_data_path = args.pheno_path.split(' ')[0]   
                pheno_test_data_path = args.pheno_path.split(' ')[1]

            ### Train
            prsice_train_cmd = 'Rscript ~/PRSice/PRSice.R --prsice ~/PRSice/PRSice_linux --base '+base_data_path+' --target '+target_train_data_path+' --binary-target '+str(binary_target)+' --pheno '+pheno_train_data_path+' --stat BETA --beta --keep-ambig --no-clump --out data/'+args.model_flag+'/'+fname_root+'_PRS_prsice_train_'+str(itr)
            ### Test
            prsice_test_cmd = 'Rscript ~/PRSice/PRSice.R --prsice ~/PRSice/PRSice_linux --base '+base_data_path+' --target '+target_test_data_path+' --binary-target '+str(binary_target)+' --pheno '+pheno_test_data_path+' --stat BETA --beta --keep-ambig --no-clump --out data/'+args.model_flag+'/'+fname_root+'_PRS_prsice_test_'+str(itr)

            # Shell calls
            subprocess.call(prsice_train_cmd, shell=True)
            subprocess.call(prsice_test_cmd, shell=True)
            

        if bool(int(args.step3)): 

            if args.realdata_directory is None:
                # Load PRS and pheno
                # Load train and test PRS
                plink_prs_train = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_prsice_train_'+str(itr)+'.best'), sep = '\s+')
                plink_prs_test = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_prsice_test_'+str(itr)+'.best'), sep = '\s+')
                prs_train = plink_prs_train.PRS.values
                prs_test = plink_prs_test.PRS.values

                # add column with ancestry name - later implement flag to choose ancestry supervise or unsupervised!
                pop_ids_train = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_train_pop_ids_'+str(itr)+'.txt'),header = None)
                pop_ids_test = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_test_pop_ids_'+str(itr)+'.txt'),header = None)
                pop_ids_train = pop_ids_train.values
                pop_ids_test = pop_ids_test.values

                # Load pheno and keep as y
                pheno_train  = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_train_'+str(itr)+'.fam'), sep = '\t', header = None)
                pheno_test  = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_test_'+str(itr)+'.fam'), sep = '\t', header = None)
                y_train = pheno_train.iloc[:,5].values
                y_train = np.expand_dims(y_train,1)
                y_test = pheno_test.iloc[:,5].values
                y_test = np.expand_dims(y_test,1)

                # Load pcs 
                pcs_train = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_train_PCA_out_'+str(itr)+'_singularVectors.txt'), sep = '\t')
                pcs_train = pcs_train.values[:,1:]

                pcs_test = pd.read_csv(str('data/'+args.model_flag+'/'+fname_root+'_PRS_test_PCA_out_'+str(itr)+'_singularVectors.txt'), sep = '\t')
                pcs_test = pcs_test.values[:,1:]

                # Split train into enviornments
                prs_train_chunks = np.array_split(prs_train,int(args.envs_flag))
                pop_ids_train_chunks = np.array_split(pop_ids_train,int(args.envs_flag))
                pcs_train_chunks = np.array_split(pcs_train,int(args.envs_flag))
                y_train_chunks = np.array_split(y_train,int(args.envs_flag)) 
                
                if itr%10 == 0:
                    print('Train and Test data successfully read in')
                
                ## Create dataloaders - train_n and test
                # Load and split into train test sets 
                train_datasets = []
                for i in range(int(args.envs_flag)):
                    train_datasets.append(PRS(prs_train_chunks[i],y_train_chunks[i],pcs_train_chunks[i],pop_ids_train_chunks[i]))

                all_train_data = PRS(prs_train,y_train,pcs_train,pop_ids_train)
                test_data = PRS(prs_test, y_test, pcs_test, pop_ids_test)

            # if real PRS data with only one input file -> read and split into train/val/test
            if args.realdata_directory is not None:
                # Real data should be a data frame with eids, prs, pheno, ancs, :covariates
                real_data_df = pd.read_csv(args.realdata_directory)
                # Need to update mapping from str to int ancestries in an automatic process
                # Quick sketch look as follows:
                # np.unique(real_data_df['ancs'], return values)
                # dict(zip([uniques],range(num_unique)))
                real_data_df['ancs'] = real_data_df['ancs'].map(dict(zip(['White','Mixed','Asian','Black', 'Not Known', 'Mxd'],[1,2,3,4,5,6])))

                # X_train, X_test, y_train, y_test = train_test_split(prs_real_full[:,:2], prs_real_full[:,-1], test_size=0.10, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(real_data_df.loc[:,real_data_df.columns != 'pheno'], real_data_df['pheno'], test_size=0.10, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.22, random_state=42)

                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()
                y_train = y_train.to_numpy()
                y_test = y_test.to_numpy()
                X_val = X_val.to_numpy()
                y_val = y_val.to_numpy()

                # Expand dims to Pytorch problems down the line
                pop_ids_train = np.expand_dims(X_train[:,2],1)
                y_train = np.expand_dims(y_train,1)
                pop_ids_val = np.expand_dims(X_val[:,2],1)
                y_val = np.expand_dims(y_val,1)
                pop_ids_test = np.expand_dims(X_test[:,2],1)
                y_test = np.expand_dims(y_test,1)
                
                num_pcs_incl = int(args.num_pcs) #Just use as number of covariates to use 
                pcs_train = X_train[:,3:(3+num_pcs_incl)]
                pcs_val = X_val[:,3:(3+num_pcs_incl)]
                pcs_test = X_test[:,3:(3+num_pcs_incl)]

                # Split train into enviornments
                prs_train_chunks = np.array_split(X_train[:,1],int(args.envs_flag))
                pop_ids_train_chunks = np.array_split(pop_ids_train,int(args.envs_flag))
                y_train_chunks = np.array_split(y_train,int(args.envs_flag)) 
                pcs_train_chunks = np.array_split(pcs_train,int(args.envs_flag))

                train_datasets = []
                for i in range(int(args.envs_flag)):
                    train_datasets.append(PRS(prs_train_chunks[i],y_train_chunks[i],pcs_train_chunks[i],pop_ids_train_chunks[i]))
                
                all_train_data = PRS(X_train[:,1],y_train, pcs_train, pop_ids_train)
                val_data = PRS(X_val[:,1], y_val, pcs_val, pop_ids_val)
                test_data = PRS(X_test[:,1], y_test, pcs_test, pop_ids_test)

            
        if bool(int(args.plot_input_dists)):
            # Create model specific results directory     
            if not os.path.isdir(str('./results/'+args.model_flag)):
                os.system(str('mkdir results/'+args.model_flag))
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Plotting PRS distributions <<<<<<<<<<<<<<<<<<<<<<')

            if itr < 1:
                if not os.path.isdir(str('./results/'+args.model_flag)):
                    os.system(str('mkdir results/'+args.model_flag))
                if not os.path.isdir(str('./results/'+args.model_flag+'/plots')):
                    os.system(str('mkdir results/'+args.model_flag+'/plots'))
            if args.realdata_directory is None:
                # Set population id file names for loading
                popids_train_fname = fname_root+'_PRS_train_'+str(itr)
                popids_test_fname = fname_root+'_PRS_test_'+str(itr)
                # Plot PRS train and test distributions by ancestry (population id) group
                plot_prs_dist(str(fname_root+'PRS_prsice_train_'+str(itr)+'.best'),popids_train_fname, path=str('data/'+args.model_flag), mod_name=args.model_flag, simrel=1)
                plot_prs_dist(str(fname_root+'PRS_prsice_test_'+str(itr)+'.best'),popids_test_fname, path=str('data/'+args.model_flag), mod_name=args.model_flag, simrel=1)
            if args.realdata_directory is not None:
                # Plot PRS dist for real data all 
                popids_df_train = pd.DataFrame(X_train[:,2])
                popids_df_train.iloc[:,0] = popids_df_train.iloc[:,0].map(dict(zip([1,2,3,4,5,6],['White','Mixed','Asian','Black', 'Not Known', 'Mxd'])))
                popids_df_test = pd.DataFrame(X_test[:,2])
                popids_df_test.iloc[:,0] = popids_df_test.iloc[:,0].map(dict(zip([1,2,3,4,5,6],['White','Mixed','Asian','Black', 'Not Known', 'Mxd'])))
                PRS_df_train = pd.DataFrame(X_train[:,1])
                PRS_df_train.columns = ['PRS']
                PRS_df_test = pd.DataFrame(X_test[:,1])
                PRS_df_test.columns = ['PRS']
                plot_fname_train = fname_root+'_PRS_train'+'_'+str(itr)
                plot_fname_test = fname_root+'_PRS_test'+'_'+str(itr)
                plot_prs_dist(popids_df=popids_df_train,PRS_df = PRS_df_train, fname_mod = plot_fname_train, mod_name=args.model_flag, simrel=0)
                plot_prs_dist(popids_df=popids_df_test,PRS_df = PRS_df_test, fname_mod = plot_fname_test, mod_name=args.model_flag, simrel=0)
            print('Saved KDE plots in results directory')
                  
        
        # Automatically set model training to true if loading data
        if bool(int(args.model_training)):
            if args.gpu == 1:
                device = 'gpu'
                gpu_num = 1
            else:
                device = 'cpu'
                gpu_num = 0
            #### Run model over user specified hyperparameters ####
            if bool(int(args.autotunning_flag)):
                ### Run FairPRS autotunning ###
                print('>>>>>>>>>>>>>>>>>>>>>> Autotuning and Evaluating FairPRS <<<<<<<<<<<<<<<<<<<<<<')
                # Run training, hyperparam search and testing for ERM and IRM
                results_dict_erm = ht.tuner(all_train_data, test_data, irm=False, trait = int(args.trait_flag), num_pcs = int(args.num_pcs), device = device, gpus_per_trial=gpu_num)

                results_dict_irm = ht.tuner(train_datasets, test_data, all_train_data = all_train_data, irm=True, num_envs = int(args.envs_flag), trait = int(args.trait_flag), num_pcs = int(args.num_pcs), device = device, gpus_per_trial=gpu_num)

                results_erm_tun[itr] = results_dict_erm
                results_irm_tun[itr] = results_dict_irm
                
                # Plot predicted PRS distributions per population
                ancs = pd.DataFrame(results_dict_irm['data']['ancs_test'])
                if args.realdata_directory is not None:
                    ancs.iloc[:,0] = ancs.iloc[:,0].map(dict(zip([1,2,3,4,5,6],['White','Mixed','Asian','Black', 'Not Known', 'Mxd'])))
                PRS_pred = pd.DataFrame(results_dict_irm['data']['PRS_pred_test'])
                PRS_pred.columns = ['PRS']
                plot_fname = fname_root+'_PRS_test_predicted'+'_'+str(itr)
                if args.realdata_directory is not None:
                    plot_prs_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=0)
                else:
                    plot_prs_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=1)

                # Plot predicted PRS distributions per population (training env 1)
                ancs = pd.DataFrame(results_dict_irm['data']['ancs_train'])
                if args.realdata_directory is not None:
                    ancs.iloc[:,0] = ancs.iloc[:,0].map(dict(zip([1,2,3,4,5,6],['White','Mixed','Asian','Black','Not Known', 'Mxd'])))
                PRS_pred = pd.DataFrame(results_dict_irm['data']['PRS_pred_train'])
                PRS_pred.columns = ['PRS']
                plot_fname = fname_root+'_PRS_train_predicted'+'_'+str(itr)
                if args.realdata_directory is not None:
                    plot_prs_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=0)
                else:
                    plot_prs_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=1)

                # Plot predicted Phenotype distributions per population
                if bool(int(args.plot_output_dists)):
                    ancs = pd.DataFrame(results_dict_irm['data']['ancs_test'])
                    if args.realdata_directory is not None:
                        ancs.iloc[:,0] = ancs.iloc[:,0].map(dict(zip([1,2,3,4,5,6],['White','Mixed','Asian','Black','Not Known', 'Mxd'])))
                    PRS_pred = pd.DataFrame(results_dict_irm['data']['Pheno_og_test'])
                    PRS_pred.columns = ['Phenotype']
                    plot_fname = fname_root+'_Phenotype_test'+'_'+str(itr)
                    if args.realdata_directory is not None:
                        plot_pheno_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=0)
                    else:
                        plot_pheno_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=1)

                    # Plot predicted Phenotype distributions per population Test
                    ancs = pd.DataFrame(results_dict_irm['data']['ancs_test'])
                    if args.realdata_directory is not None:
                        ancs.iloc[:,0] = ancs.iloc[:,0].map(dict(zip([1,2,3,4,5,6],['White','Mixed','Asian','Black','Not Known', 'Mxd'])))
                    PRS_pred = pd.DataFrame(results_dict_irm['data']['Pheno_pred_test'])
                    PRS_pred.columns = ['Phenotype']
                    plot_fname = fname_root+'_Phenotype_test_predicted'+'_'+str(itr)
                    if args.realdata_directory is not None:
                        plot_pheno_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=0)
                    else:
                        plot_pheno_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = plot_fname, mod_name=args.model_flag, simrel=1)    
        

            # Display results if there was a model training procedure
            if (itr+1)%10 == 0:
                print(f'{i} iterations completed.')
            # Create model specific results directory     
            if not os.path.isdir(str('./results/'+args.model_flag)):
                os.system(str('mkdir results/'+args.model_flag))

            print('='*40)
            # Load results from all iterations for summary results                 
            temp_erm = []
            for k in results_erm_tun.keys():
                temp_erm.append(pd.DataFrame.from_dict(results_erm_tun[k]['metrics']))

            #  Save ERM summary results
            metrics_df_erm = pd.concat(temp_erm)
            metrics_df_erm_out = pd.DataFrame(metrics_df_erm.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T
            metrics_df_erm_out.to_csv(str('results/'+args.model_flag+'/'+fname_root+'_ERM_iters_results_'+str(itr)+'.csv'),index=False)

            # Save dictionary of all reults, data, and hyperparams used per iteration on ERM
            with open(str('results/'+args.model_flag+'/'+fname_root+'_ERM_iters_results_dictionary_'+str(itr)+'.pkl'), 'wb') as f:
                pickle.dump(results_erm_tun, f)

            # Display results for ERM 
            print('Results saved into ',str('results/'+args.model_flag+'/'+fname_root+'_ERM_iters_results_'+str(itr)+'.csv'))
            print(f'Average ERM model results: train MSE {round(metrics_df_erm_out["loss_train"][0],5)}, test MSE {round(metrics_df_erm_out["loss_test"][0],5)}, R2 original PRS {round(metrics_df_erm_out["R2_og_test"][0],5)}, R2 predicted PRS {round(metrics_df_erm_out["R2_pred_test"][0],5)}, r2 prs {round(metrics_df_erm_out["r2_prs_test"][0],5)}, r2 pheno {round(metrics_df_erm_out["r2_pheno_test"][0],5)}')
            
            print('='*40)
            
            # IRM
            temp = []
            for k in results_irm_tun.keys():
                temp.append(pd.DataFrame.from_dict(results_irm_tun[k]['metrics']))
                
            metrics_df = pd.concat(temp)
            metrics_df_out = pd.DataFrame(metrics_df.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T
            metrics_df_out.to_csv(str('results/'+args.model_flag+'/'+fname_root+'_IRM_'+str(itr)+'_iters_results.csv'),index=False)
            pd.DataFrame(metrics_df_out.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T.to_csv(str('results_/'+args.model_flag+'/'+fname_root+'_IRM_'+str(itr)+'_iters_results.csv'),index=False)

            # Save dictionary of all results, data, and hyperparams used per iteration on IRM
            with open(str('results/'+args.model_flag+'/IRM_'+fname_root+'_'+str(itr)+'_iters_results_dictionary.pkl'), 'wb') as f:
                pickle.dump(results_irm_tun, f)

            # Display results for IRM 
            print('Results saved into ',str('results/'+args.model_flag+'/IRM_'+fname_root+'_'+str(itr)+'_iters_results.csv'))
            print(f'Average IRM model results: train MSE {round(metrics_df_out["loss_train"][0],5)}, test MSE {round(metrics_df_out["loss_test"][0],5)}, R2 original PRS {round(metrics_df_out["R2_og_test"][0],5)}, R2 predicted PRS {round(metrics_df_out["R2_pred_test"][0],5)}, r2 prs {round(metrics_df_out["r2_prs_test"][0],5)},r2 pheno {round(metrics_df_out["r2_pheno_test"][0],5)}')
            print('='*40)
    
    print("--- Total run time in %s seconds ---" % (time.time() - begin_time))