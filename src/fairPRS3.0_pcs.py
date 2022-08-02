import os, datetime, random, sys, shutil, itertools, math, argparse, time, subprocess, re, string
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd
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
import hyperparam_tuning_pcs_losses as ht_losses
import legacy_train_test as leg_methods

def msg(name=None):
    return '''CluStrat_wrapper.py
         >> python3 CluStrat_wrapper.py --sim 1 --prop 10,20,70 --trait 1 --model BN --ver 0 --plot 0 --pval 0.0001 --numclust 10 --size 1000,1000
         >> python3 CluStrat_wrapper.py --dir /testing/test_data --pval 0.0001 --numclust 10
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
    parser.add_argument("-sim_data", "--simulate_data", dest='sim_data', action='store', help='Do you want to simulate the datasets?', metavar='SIMDATA')
    parser.add_argument("-prs_path", "--PRS_load_path", dest='PRS_load_path', action='store', help='Enter the path to your PRS files', metavar='PRSPATH')
    parser.add_argument("-model_train", "--model_training", dest='model_training', action='store', help='Do you want to trainany models? 1-yes 0-no', metavar='MODELTRAIN')
    parser.add_argument("-plot_inp_dist", "--plot_input_distributions", dest='plot_input_dists', action='store', help='Do you want to plot the PRS distributions of input data? 1- yes  0- no', metavar='PLOTINPUT')
    parser.add_argument("-train_fname", "--train_data_file_name", dest='train_fname', action='store', help='Enter training PRS file name (must be stored in data/model/fname', metavar='TRAINFNAME')
    parser.add_argument("-test_fname", "--test_data_file_name", dest='test_fname', action='store', help='Enter test PRS file name (must be stored in data/model/fname)', metavar='TESTFNAME')
    parser.add_argument("-fname_root", "--fname_root_outfiles", dest='fname_root', action='store', help='Enter root/common name for outfiles', metavar='FNAMEROOT')
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
        pop_ids = pd.read_csv(str(path+'/pop_ids_'+popids_fname+'.txt'),header = None)
    else:
        pop_ids = popids_df
    if PRS_df is None:
        PRS = pd.read_csv(str(path+'/'+prs_fname),sep = '\s+')
    else:
        PRS = PRS_df
    PRS['pop_id'] = pop_ids.values
    if simrel ==1:
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
        plt.savefig(str('results_wpcs/'+mod_name+'/plots/'+popids_fname+'.png'))
    else:
        plt.savefig(str('results_wpcs/'+mod_name+'/plots/'+fname_mod+'.png'))
    plt.clf()
def plot_pheno_dist(prs_fname = None ,popids_fname= None, PRS_df = None, popids_df = None, fname_mod = None, path = 'data/', mod_name=None, simrel=None):
    sns.set_style('darkgrid')
    if popids_df is None:
        pop_ids = pd.read_csv(str(path+'/pop_ids_'+popids_fname+'.txt'),header = None)
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
        plt.savefig(str('results_wpcs/'+mod_name+'/plots/'+popids_fname+'.png'))
    else:
        plt.savefig(str('results_wpcs/'+mod_name+'/plots/'+fname_mod+'.png'))
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
    if not os.path.isdir('./scripts'):
        print('>>>>>>>>>>>>>>>>>> Preparing environment <<<<<<<<<<<<<<<<<<<<<<')
        os.system('mkdir data')
        os.system('mkdir scripts')
        os.system('mkdir results_wpcs')
        os.system('mv *.py ./scripts')
        os.system('mv ./scripts/fairPRS.py .')
    os.system('cd scripts')
    
    
    if int(args.trait_flag) == 1:
        pheno = 'continuous'
        binary_target = 'F'
    elif int(args.trait_flag) == 0:
        pheno = 'Binary'
        binary_target = 'T'
    
    iters = int(args.iters)

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
    results_erm = {'train_mse':[], 'test_mse':[], 'R2_og': [],'R2_pred':[], 'r2':[], 'optimal':[] ,'penalty':[], 'lr':[], 'zt_med': [], 'u1':[], 'u2':[],'u3':[],'u4':[],'u5':[], 'prs_pred_train':[], 'prs_pred_test':[], 'ancs_train':[], 'ancs_test':[], 'pheno_train': [], 'pheno_test': [], 'targets_train':[],'targets_test':[],'prs_og_train':[],'prs_og_test':[]}
    results_IRM = {'train_mse':[], 'test_mse':[], 'R2_og': [],'R2_pred':[], 'r2':[], 'optimal':[] ,'penalty':[], 'lr':[], 'zt_med': [], 'u1':[], 'u2':[],'u3':[],'u4':[],'u5':[], 'prs_pred_train':[], 'prs_pred_test':[], 'ancs_train':[], 'ancs_test':[], 'pheno_train': [], 'pheno_test': [], 'targets_train':[],'targets_test':[],'prs_og_train':[],'prs_og_test':[]}
    
    results_erm_tun = {}
    results_irm_tun = {}

    res_metrics_erm = np.empty((iters,9)) # train_loss, train_R2_pred, train_corr_prs, train_corr_pehno, R2_og, val_loss, test_R2_pred, test_corr_prs, test_corr_pheno
    res_metrics_irm = np.empty((iters,9))

    # Create directory for corresponding model if it doesn't exist
    if not os.path.isdir(str('./data/'+args.model_flag)):
        os.system(str('mkdir data/'+args.model_flag))
            
    for itr in range(iters):
        if bool(int(args.sim_data)):
            ### Create sim data ###
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Simulating data <<<<<<<<<<<<<<<<<<<<<<')
            # Train data (GWAS - Betas)  
#             sim_train_cmd = 'python scripts/data_simulate.py --model '+str(args.model_flag)+' --prop '+str(prop_genetic)+','+str(prop2_environmental)+','+str(prop3_noise)+' --pheno '+str(pheno)+' --size '+str(train_size)+','+str(args.num_snps_flag)+' -causal '+str(args.num_causal)+' -num_pop '+str(args.num_pop)+' --path data/'+args.model_flag+'/ -fname_mod train_'+str(itr)+' --random_seed '+str(args.random_seed)
            sim_train_cmd = 'python scripts/data_simulate_risk.py --model '+str(args.model_flag)+' --prop '+str(prop_genetic)+','+str(prop2_environmental)+','+str(prop3_noise)+' --pheno '+str(pheno)+' --size '+str(train_size)+','+str(args.num_snps_flag)+' -causal '+str(args.num_causal)+' -num_pop '+str(args.num_pop)+' --path data/'+args.model_flag+'/ -fname_mod train_'+str(itr)+' --random_seed '+str(args.random_seed)+ ' -r '+args.risk_flag

            # Train data  (PRS)
#             prs_train_cmd = 'python scripts/data_simulate.py --model '+str(args.model_flag)+' --prop '+str(prop_genetic)+','+str(prop2_environmental)+','+str(prop3_noise)+' --pheno '+str(pheno)+' --size '+str(PRS_train_size)+','+str(args.num_snps_flag)+' -causal '+str(args.num_causal)+' -num_pop '+str(args.num_pop)+' --path data/'+args.model_flag+'/ -fname_mod PRS_train_'+str(itr)+' --random_seed '+str(args.random_seed)
            prs_train_cmd = 'python scripts/data_simulate_risk.py --model '+str(args.model_flag)+' --prop '+str(prop_genetic)+','+str(prop2_environmental)+','+str(prop3_noise)+' --pheno '+str(pheno)+' --size '+str(PRS_train_size)+','+str(args.num_snps_flag)+' -causal '+str(args.num_causal)+' -num_pop '+str(args.num_pop)+' --path data/'+args.model_flag+'/ -fname_mod PRS_train_'+str(itr)+' --random_seed '+str(args.random_seed)+ ' -r '+args.risk_flag

            # Test data  (PRS)
#             prs_test_cmd = 'python scripts/data_simulate.py --model '+str(args.model_flag)+' --prop '+str(prop_genetic)+','+str(prop2_environmental)+','+str(prop3_noise)+' --pheno '+str(pheno)+' --size '+str(PRS_test_size)+','+str(args.num_snps_flag)+' -causal '+str(args.num_causal)+' -num_pop '+str(args.num_pop)+' --path data/'+args.model_flag+'/ -fname_mod PRS_test_'+str(itr)+' --random_seed '+str(args.random_seed)
            prs_test_cmd = 'python scripts/data_simulate_risk.py --model '+str(args.model_flag)+' --prop '+str(prop_genetic)+','+str(prop2_environmental)+','+str(prop3_noise)+' --pheno '+str(pheno)+' --size '+str(PRS_test_size)+','+str(args.num_snps_flag)+' -causal '+str(args.num_causal)+' -num_pop '+str(args.num_pop)+' --path data/'+args.model_flag+'/ -fname_mod PRS_test_'+str(itr)+' --random_seed '+str(args.random_seed)+ ' -r '+args.risk_flag

            # Simulation Calls
            subprocess.call(sim_train_cmd, shell=True)
            subprocess.call(prs_train_cmd, shell=True)
            subprocess.call(prs_test_cmd, shell=True)

            # Set up common name root for all files
            fname_root = str(args.model_flag)+'_'+str(prop_genetic)+'_'+str(prop2_environmental)+'_'+str(prop3_noise)+'_'+str(pheno)+'_'+str(args.num_causal)

            ### Compute GWAS ###
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Computing GWAS <<<<<<<<<<<<<<<<<<<<<<')
            # Compute PCS / covariates
            tera_pca_train_cmd = '../TeraPCA/TeraPCA.exe -bfile data/'+args.model_flag+'/simfile_'+fname_root+'_train_'+str(itr)+' -nsv '+args.num_pcs+' -filewrite 1 -prefix data/'+args.model_flag+'/PCA_out_'+fname_root+'_train_'+str(itr)
            tera_pca_prs_train_cmd = '../TeraPCA/TeraPCA.exe -bfile data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_train_'+str(itr)+' -nsv '+args.num_pcs+' -filewrite 1 -prefix data/'+args.model_flag+'/PCA_out_'+fname_root+'_PRS_train_'+str(itr)
            tera_pca_prs_test_md = '../TeraPCA/TeraPCA.exe -bfile data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_test_'+str(itr)+' -nsv '+args.num_pcs+' -filewrite 1 -prefix data/'+args.model_flag+'/PCA_out_'+fname_root+'_PRS_test_'+str(itr)
            subprocess.call(tera_pca_train_cmd, shell=True)
            subprocess.call(tera_pca_prs_train_cmd, shell=True)
            subprocess.call(tera_pca_prs_test_md, shell=True)
            #Load covariates and fit to plink format - covariate
            cov = pd.read_csv(str('data/'+args.model_flag+'/PCA_out_'+fname_root+'_train_'+str(itr)+'_singularVectors.txt'), sep='\t')
            cov.insert(1, 'IID', cov['FID'])
            cov_np = cov.to_numpy(dtype='str')
            cov_out = np.insert(cov_np,0,cov.columns.values, axis =0)
            # cov.to_csv(str('data/'+args.model_flag+'/PCA_out_'+fname_root+'_covariates.txt'),index=False)
            np.savetxt(str('data/'+args.model_flag+'/PCA_out_'+fname_root+'_covariates_'+str(itr)+'.cov'), cov_out, delimiter='\t', fmt='%s')

            # Run GWAS with plink GLM
            gwas_cmd = 'plink2 --bfile data/'+args.model_flag+'/simfile_'+fname_root+'_train_'+str(itr)+' --glm --out data/'+args.model_flag+'/GLM_results_'+fname_root+'_'+str(itr)+' --covar data/'+args.model_flag+'/PCA_out_'+fname_root+'_covariates_'+str(itr)+'.cov'
            subprocess.call(gwas_cmd, shell=True)

            # Update GLM (GWAS) headers for standard use on PRS software and Manhattan plotting library
            nam = 'data/'+args.model_flag+'/GLM_results_'+fname_root+'_'+str(itr)+'.PHENO1.glm.linear'
            GLM = pd.read_csv(nam, sep = '\s+')
            GLM.columns = ['CHR', 'BP', 'SNP', 'A2', 'ALT','A1', 'TEST','OBS_CT', 'BETA', 'SE', 'T_STAT', 'P']
            # Drop all duplicated SNPs / only leave values where TEST column = ADD (Band aid fix)
            GLM = GLM.loc[GLM['TEST'] =='ADD']
            # Filling Na values with column means (band aid fix check later cause) modifed 2022.07.26
            GLM['BETA'] = GLM['BETA'].fillna(GLM['BETA'].mean())
            GLM['SE'] = GLM['SE'].fillna(GLM['SE'].mean())
            GLM['T_STAT'] = GLM['T_STAT'].fillna(GLM['T_STAT'].mean())
            GLM['P'] = GLM['P'].fillna(GLM['P'].mean())
            GLM.to_csv(str('data/'+args.model_flag+'/GLM_DF_'+fname_root+'_'+str(itr)+'.txt'), sep='\t', index = False)

            # Plot GWAS - Manhattan plot
        #     print('>>>>>>>>>>>>>>>>>> Plotting Manhattan plot for GWAS <<<<<<<<<<<<<<<<<<<<<<')

        #     ## Need to find alternative way to qmplot, takes too long to ouput image !!!
        #     nam_k = str(prop_genetic)+'_'+str(prop2_environmental)+'_'+str(prop3_noise)+'_'+str(pheno)
        #     man_plot_cmd = 'qmplot -I '+nam+' -T '+nam_k+' --dpi 300 -O '+'results_wpcs/'+nam_k
        #     subprocess.call(man_plot_cmd, shell=True)
        #     print('>>>> Saved on Results folder <<<<<')

            ### Compute PRS ###
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Computing PRS <<<<<<<<<<<<<<<<<<<<<<')

            # Calculate train and test PRS
            ## Intermediate file (phenotype) for PRS calculation
            ### Train
            pheno_out = pd.read_csv('data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_train_'+str(itr)+'.fam', sep = '\s+', header = None)
            pheno_out = pheno_out.iloc[:,[0,1,5]]
            pheno_out.columns  = ['FID', 'IID', 'PHENO']
            pheno_out.to_csv(str('data/'+args.model_flag+'/pheno_out_train_'+fname_root+'_'+str(itr)+'.txt'), sep='\t', index = False)
            ### Test
            pheno_out = pd.read_csv('data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_test_'+str(itr)+'.fam', sep = '\s+', header = None)
            pheno_out = pheno_out.iloc[:,[0,1,5]]
            pheno_out.columns  = ['FID', 'IID', 'PHENO']
            pheno_out.to_csv(str('data/'+args.model_flag+'/pheno_out_test_'+fname_root+'_'+str(itr)+'.txt'), sep='\t', index = False)

            ## PRSice commands - Note: change PRSice location to relaive path!
            ### Train
            prsice_train_cmd = 'Rscript ~/PRSice/PRSice.R --prsice ~/PRSice/PRSice_linux --base data/'+args.model_flag+'/GLM_DF_'+fname_root+'_'+str(itr)+'.txt --target data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_train_'+str(itr)+' --binary-target '+str(binary_target)+' --pheno data/'+args.model_flag+'/pheno_out_train_'+fname_root+'_'+str(itr)+'.txt --stat BETA --beta --keep-ambig --no-clump --out data/'+args.model_flag+'/PRS_prsice_train_'+fname_root+'_'+str(itr)

            ### Test
            prsice_test_cmd = 'Rscript ~/PRSice/PRSice.R --prsice ~/PRSice/PRSice_linux --base data/'+args.model_flag+'/GLM_DF_'+fname_root+'_'+str(itr)+'.txt --target data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_test_'+str(itr)+' --binary-target '+str(binary_target)+' --pheno data/'+args.model_flag+'/pheno_out_test_'+fname_root+'_'+str(itr)+'.txt --stat BETA --beta --keep-ambig --no-clump --out data/'+args.model_flag+'/PRS_prsice_test_'+fname_root+'_'+str(itr)

            # Shell calls
            subprocess.call(prsice_train_cmd, shell=True)
            subprocess.call(prsice_test_cmd, shell=True)
            


            ## Load PRS and pheno
            # Load train and test PRS
            plink_prs_train = pd.read_csv(str('data/'+args.model_flag+'/PRS_prsice_train_'+fname_root+'_'+str(itr)+'.best'), sep = '\s+')
            plink_prs_test = pd.read_csv(str('data/'+args.model_flag+'/PRS_prsice_test_'+fname_root+'_'+str(itr)+'.best'), sep = '\s+')
            prs_train = plink_prs_train.PRS.values
            prs_test = plink_prs_test.PRS.values

            # add column with ancestry name - later implement flag to choose ancestry supervise or unsupervised!
        #     fname_root = str(args.model_flag)+'_'+prop_genetic+'_'+prop2_environmental+'_'+prop3_noise+'_'+pheno
            popids_fname = fname_root+'_PRS_train'
            pop_ids_train = pd.read_csv(str('data/'+args.model_flag+'/pop_ids_'+popids_fname+'_'+str(itr)+'.txt'),header = None)
            pop_ids_train = pop_ids_train.values

            popids_fname = fname_root+'_PRS_test'
            pop_ids_test = pd.read_csv(str('data/'+args.model_flag+'/pop_ids_'+popids_fname+'_'+str(itr)+'.txt'),header = None)
            pop_ids_test = pop_ids_test.values

            # Load pheno and keep as y
            pheno_train  = pd.read_csv(str('data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_train_'+str(itr)+'.fam'), sep = '\t', header = None)
            pheno_test  = pd.read_csv(str('data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_test_'+str(itr)+'.fam'), sep = '\t', header = None)
            y_train = pheno_train.iloc[:,5].values
            y_train = np.expand_dims(y_train,1)
            y_test = pheno_test.iloc[:,5].values
            y_test = np.expand_dims(y_test,1)

            # Load pcs 
            pcs_fname = fname_root+'_PRS_train'
            pcs_train = pd.read_csv(str('data/'+args.model_flag+'/PCA_out_'+pcs_fname+'_'+str(itr)+'_singularVectors.txt'), sep = '\t')
            pcs_train = pcs_train.values
            pcs_train = pcs_train[:,1:]

            pcs_fname = fname_root+'_PRS_test'
            pcs_test = pd.read_csv(str('data/'+args.model_flag+'/PCA_out_'+pcs_fname+'_'+str(itr)+'_singularVectors.txt'), sep = '\t')
            pcs_test = pcs_test.values
            pcs_test = pcs_test[:,1:]

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
        
        if load_data:
            if args.train_fname is not None:
                ## Load PRS and pheno
                # Load train and test PRS
                # plink_prs = pd.read_csv(str('data/PRS_prsice_train_'+fname_root+'.best'), sep = '\s+')
                # fname_root = args.train_fname.split('/')[-1]
                # fname_root = '_'.join(args.fname_root.split('_')[3:9])
                fname_root = '_'.join(args.train_fname.split('_')[3:9])
                plink_prs_train = pd.read_csv(str('data/'+args.model_flag+'/PRS_prsice_train_'+fname_root+'_'+str(itr)+'.best'), sep = '\s+')
                plink_prs_test = pd.read_csv(str('data/'+args.model_flag+'/PRS_prsice_test_'+fname_root+'_'+str(itr)+'.best'), sep = '\s+')
                prs_train = plink_prs_train.PRS.values
                prs_test = plink_prs_test.PRS.values

                # add column with ancestry name - later implement flag to choose ancestry supervise or unsupervised!
            #     fname_root = str(args.model_flag)+'_'+prop_genetic+'_'+prop2_environmental+'_'+prop3_noise+'_'+pheno
                # popids_fname = fname_root+'_PRS_train'
                # pop_ids = pd.read_csv(str('data/pop_ids_'+popids_fname+'.txt'),header = None)
                pop_ids_train = pd.read_csv(str('data/'+args.model_flag+'/pop_ids_'+fname_root+'_PRS_train_'+str(itr)+'.txt'),header = None)
                pop_ids_test = pd.read_csv(str('data/'+args.model_flag+'/pop_ids_'+fname_root+'_PRS_test_'+str(itr)+'.txt'),header = None)
                pop_ids_train = pop_ids_train.values
                pop_ids_test = pop_ids_test.values

                # Load pheno and keep as y
                
                pheno_train  = pd.read_csv(str('data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_train_'+str(itr)+'.fam'), sep = '\t', header = None)
                pheno_test  = pd.read_csv(str('data/'+args.model_flag+'/simfile_'+fname_root+'_PRS_test_'+str(itr)+'.fam'), sep = '\t', header = None)
                y_train = pheno_train.iloc[:,5].values
                y_train = np.expand_dims(y_train,1)
                y_test = pheno_test.iloc[:,5].values
                y_test = np.expand_dims(y_test,1)

                # Load pcs 
                pcs_fname = fname_root+'_PRS_train'
                pcs_train = pd.read_csv(str('data/'+args.model_flag+'/PCA_out_'+pcs_fname+'_'+str(itr)+'_singularVectors.txt'), sep = '\t')
                pcs_train = pcs_train.values
                pcs_train = pcs_train[:,1:]

                pcs_fname = fname_root+'_PRS_test'
                pcs_test = pd.read_csv(str('data/'+args.model_flag+'/PCA_out_'+pcs_fname+'_'+str(itr)+'_singularVectors.txt'), sep = '\t')
                pcs_test = pcs_test.values
                pcs_test = pcs_test[:,1:]

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
                real_data_df = pd.read_csv(args.realdata_directory)
                # if isinstance(real_data_df['ancs'].values[0],int) or isinstance(real_data_df['ancs'].values[0],float):
                real_data_df['ancs'] = real_data_df['ancs'].map(dict(zip(['White','Mixed','Asian','Black', 'Not Known', 'Mxd'],[1,2,3,4,5,6])))

                # prs_real_full = np.stack((real_prs,real_ancs,real_pheno)).T

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
                
                num_pcs_incl = int(args.num_pcs)
                pcs_train = X_train[:,3:(3+3+num_pcs_incl)]
                pcs_val = X_val[:,3:(3+3+num_pcs_incl)]
                pcs_test = X_test[:,3:(3+3+num_pcs_incl)]

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
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Plotting PRS distributions <<<<<<<<<<<<<<<<<<<<<<')

            if itr < 1:
                if not os.path.isdir(str('./results_wpcs/'+args.model_flag)):
                    os.system(str('mkdir results_wpcs/'+args.model_flag))
                if not os.path.isdir(str('./results_wpcs/'+args.model_flag+'/plots')):
                    os.system(str('mkdir results_wpcs/'+args.model_flag+'/plots'))
            if args.realdata_directory is None:
                # Set population id file names for loading
                popids_train_fname = fname_root+'_PRS_train'+'_'+str(itr)
                popids_test_fname = fname_root+'_PRS_test'+'_'+str(itr)
                # Plot PRS train and test distributions by ancestry (population id) group
                plot_prs_dist(str('PRS_prsice_train_'+fname_root+'_'+str(itr)+'.best'),popids_train_fname, path=str('data/'+args.model_flag), mod_name=args.model_flag, simrel=1)
                plot_prs_dist(str('PRS_prsice_test_'+fname_root+'_'+str(itr)+'.best'),popids_test_fname, path=str('data/'+args.model_flag), mod_name=args.model_flag, simrel=1)
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
                # results_dict_erm = ht_losses.tuner(all_train_data, test_data, irm=False, trait = int(args.trait_flag), num_pcs = int(args.num_pcs), device = device, gpus_per_trial=gpu_num)

                results_dict_irm = ht.tuner(train_datasets, test_data, all_train_data = all_train_data, irm=True, num_envs = int(args.envs_flag), trait = int(args.trait_flag), num_pcs = int(args.num_pcs), device = device, gpus_per_trial=gpu_num)
                # results_dict_irm = ht_losses.tuner(train_datasets, test_data, all_train_data = all_train_data, irm=True, num_envs = int(args.envs_flag), trait = int(args.trait_flag), num_pcs = int(args.num_pcs), device = device, gpus_per_trial=gpu_num)

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

#           ###########################################################################################################            
            # No autotunning run over n iterations
            if bool(int(args.autotunning_flag)) == False:
                ## Define parameters for models
                lr_erm, lr_irm = args.lrs.split(',')
                lr_erm, lr_irm = float(lr_erm), float(lr_irm)
                pen_mult = float(args.pen_mult)
                units_erm = args.units_erm.split(',')
                units_erm = list(map(int, units_erm))
                units_irm = args.units_irm.split(',')
                units_irm = list(map(int, units_irm))
                
            
                ## ERM Baseline 
                tr_mse, R2_og_tr, R2_pred_tr, tr_corr, tr_prs_pred, tr_prs, tr_zt_med, tr_ancs, tr_pred_phenos, tr_targets, t_mse, R2_og_t, R2_pred_t, test_corr, t_prs_pred, t_prs, zt_med, t_ancs, t_pred_phenos, t_targets = leg_methods.train_and_test_erm(lr = lr_erm, u = units_erm, train_data = all_train_data, test_data = test_data)
                results_erm['train_mse'].append(tr_mse)
                results_erm['test_mse'].append(t_mse)
                results_erm['R2_og'].append(R2_og_t)
                results_erm['R2_pred'].append(R2_pred_t)
                results_erm['r2'].append(test_corr)
                results_erm['optimal'].append(False)
                results_erm['penalty'].append(False)
                results_erm['lr'].append(lr_irm)
                results_erm['zt_med'].append(zt_med)
                results_erm['prs_pred_train'].append(tr_prs_pred)
                results_erm['prs_pred_test'].append(t_prs_pred)  
                results_erm['prs_og_train'].append(tr_prs)
                results_erm['prs_og_test'].append(t_prs)
                results_erm['ancs_test'].append(t_ancs)
                results_erm['ancs_train'].append(tr_ancs)
                results_erm['pheno_train'].append(tr_pred_phenos)
                results_erm['pheno_test'].append(t_pred_phenos)
                results_erm['targets_train'].append(tr_targets)
                results_erm['targets_test'].append(t_targets)
                u_pad = np.pad(units_erm,(0,5-len(units_erm)),constant_values = 0)
                results_erm['u1'].append(u_pad[0])
                results_erm['u2'].append(u_pad[1])
                results_erm['u3'].append(u_pad[2])
                results_erm['u4'].append(u_pad[3])
                results_erm['u5'].append(u_pad[4])
                

                ## IRM
                tr_mse,R2_og_tr, R2_pred_tr, r2_tr, prs_pred_tr, zt_med_tr, ancs_tr, pheno_pred_tr, targets_tr, prs_tr, t_mse, R2_og_t, R2_pred_t, r2_t, prs_pred_t, zt_med_t, ancs_t, pheno_pred_t, targets_t, prs_t, optimal = leg_methods.train_and_test_irm(lr = lr_irm, u = units_irm, pen_mult = pen_mult, num_env= int(args.envs_flag), train_datasets = train_datasets, all_train_data=all_train_data, test_data =test_data )
                results_IRM['train_mse'].append(tr_mse)
                results_IRM['test_mse'].append(t_mse)
                results_IRM['R2_og'].append(R2_og_t)
                results_IRM['R2_pred'].append(R2_pred_t)
                results_IRM['r2'].append(r2_t)
                results_IRM['optimal'].append(optimal)
                results_IRM['penalty'].append(pen_mult)
                results_IRM['lr'].append(lr_irm)
                results_IRM['zt_med'].append(zt_med)  
                results_IRM['prs_pred_train'].append(prs_pred_tr)
                results_IRM['prs_pred_test'].append(prs_pred_t)  
                results_IRM['prs_og_train'].append(prs_tr)
                results_IRM['prs_og_test'].append(prs_t)
                results_IRM['ancs_test'].append(ancs_t)
                results_IRM['ancs_train'].append(ancs_tr)
                results_IRM['pheno_train'].append(pheno_pred_tr)
                results_IRM['pheno_test'].append(pheno_pred_t)
                results_IRM['targets_train'].append(targets_tr)
                results_IRM['targets_test'].append(targets_t)
                u_pad = np.pad(units_irm,(0,5-len(units_irm)),constant_values = 0)
                results_IRM['u1'].append(u_pad[0])
                results_IRM['u2'].append(u_pad[1])
                results_IRM['u3'].append(u_pad[2])
                results_IRM['u4'].append(u_pad[3])
                results_IRM['u5'].append(u_pad[4])
        
    
    # Display results if there was a model training procedure
    if bool(int(args.model_training)):
        if (itr+1)%10 == 0:
            print(f'{i} iterations completed.')
            

        if not os.path.isdir(str('./results_wpcs/'+args.model_flag)):
            os.system(str('mkdir results_wpcs/'+args.model_flag))
                        
    #     avg_results_erm = []
    #     for k in results_erm.keys():
    # #         print(results_erm[k][0])
    #         if isinstance(results_erm[k][0], float):
    #             avg_results_erm.append(np.mean(results_erm[k]))
        
        temp_erm = []
        for k in results_erm_tun.keys():
            temp_erm.append(pd.DataFrame.from_dict(results_erm_tun[k]['metrics']))
            
        metrics_df_erm = pd.concat(temp_erm)
        metrics_df_erm_out = pd.DataFrame(metrics_df_erm.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T
        metrics_df_erm_out.to_csv(str('results_wpcs/'+args.model_flag+'/ERM_'+fname_root+'_'+str(itr)+'_iters_results.csv'),index=False)
        # np.savetxt(str('results_wpcs/ERM_'+fname_root+'_'+str(itr)+'_iters_avg.txt'),np.transpose(np.expand_dims(np.array(avg_results_erm),axis=1)),delimiter = '\t', fmt= '%.7f')
        print('='*40)

        with open(str('results_wpcs/'+args.model_flag+'/ERM_'+fname_root+'_'+str(itr)+'_iters_results_dictionary.pkl'), 'wb') as f:
            pickle.dump(results_erm_tun, f)

        # Display results for ERM 
        print('Results saved into ',str('results_wpcs/'+args.model_flag+'/ERM_'+fname_root+'_'+str(itr)+'_iters_results.csv'))
        print(f'Average ERM model results: train MSE {round(metrics_df_erm_out["loss_train"][0],5)}, test MSE {round(metrics_df_erm_out["loss_test"][0],5)}, R2 original PRS {round(metrics_df_erm_out["R2_og_test"][0],5)}, R2 predicted PRS {round(metrics_df_erm_out["R2_pred_test"][0],5)}, r2 prs {round(metrics_df_erm_out["r2_prs_test"][0],5)}, r2 pheno {round(metrics_df_erm_out["r2_pheno_test"][0],5)}')
        
        
    #     avg_results_irm = []
    #     for k in results_IRM.keys():
    #         if isinstance(results_IRM[k][0], float):
    #             avg_results_irm.append(np.mean(results_IRM[k]))
        temp = []
        for k in results_irm_tun.keys():
            temp.append(pd.DataFrame.from_dict(results_irm_tun[k]['metrics']))
            
        metrics_df = pd.concat(temp)
        metrics_df_out = pd.DataFrame(metrics_df.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T
        metrics_df_out.to_csv(str('results_wpcs/'+args.model_flag+'/ERM_'+fname_root+'_'+str(itr)+'_iters_results.csv'),index=False)
        pd.DataFrame(metrics_df_out.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T.to_csv(str('results_wpcs/'+args.model_flag+'/IRM_'+fname_root+'_'+str(itr)+'_iters_results.csv'),index=False)
        # np.savetxt(str('results_wpcs/IRM_'+fname_root+'_'+str(itr)+'_iters_avg.txt'),np.transpose(np.expand_dims(np.array(avg_results_irm),axis=1)),delimiter = '\t', fmt= '%.7f')
        print('='*40)

        with open(str('results_wpcs/'+args.model_flag+'/IRM_'+fname_root+'_'+str(itr)+'_iters_results_dictionary.pkl'), 'wb') as f:
            pickle.dump(results_irm_tun, f)

    #     # Display results for IRM 
        print('Results saved into ',str('results_wpcs/'+args.model_flag+'/IRM_'+fname_root+'_'+str(itr)+'_iters_results.csv'))
        print(f'Average IRM model results: train MSE {round(metrics_df_out["loss_train"][0],5)}, test MSE {round(metrics_df_out["loss_test"][0],5)}, R2 original PRS {round(metrics_df_out["R2_og_test"][0],5)}, R2 predicted PRS {round(metrics_df_out["R2_pred_test"][0],5)}, r2 prs {round(metrics_df_out["r2_prs_test"][0],5)},r2 pheno {round(metrics_df_out["r2_pheno_test"][0],5)}')
 
    print("--- Test time in %s seconds ---" % (time.time() - begin_time))