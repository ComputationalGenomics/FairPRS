import subprocess
import numpy as np
import pandas as pd

def prs_base(fname_root, model_flag, gwas_path, geno_path, pheno_path, covs_path, binary_target, itr):
    ### Compute PRS ###
    if itr%10 == 0:
        print('>>>>>>>>>>>>>>>>>>>>>> Computing PRS <<<<<<<<<<<<<<<<<<<<<<')

    # Calculate train and test PRS
    ## Intermediate file (phenotype) for PRS calculation
    if pheno_path is None:
        ### Train
        pheno_out = pd.read_csv('../data/'+model_flag+'/'+fname_root+'_PRS_train_'+str(itr)+'.fam', sep = '\s+', header = None)
        pheno_out = pheno_out.iloc[:,[0,1,5]]
        pheno_out.columns  = ['FID', 'IID', 'PHENO']
        pheno_out.to_csv(str('../data/'+model_flag+'/'+fname_root+'_pheno_out_train_'+str(itr)+'.txt'), sep='\t', index = False)
        ### Test
        pheno_out = pd.read_csv('../data/'+model_flag+'/'+fname_root+'_PRS_test_'+str(itr)+'.fam', sep = '\s+', header = None)
        pheno_out = pheno_out.iloc[:,[0,1,5]]
        pheno_out.columns  = ['FID', 'IID', 'PHENO']
        pheno_out.to_csv(str('../data/'+model_flag+'/'+fname_root+'_pheno_out_test_'+str(itr)+'.txt'), sep='\t', index = False)


    ## PRSice commands - Note: change PRSice location to relaive path!
    if gwas_path is None:
        base_data_path = '../data/'+model_flag+'/'+fname_root+'_GLM_DF_'+str(itr)+'.txt'
    else:
        base_data_path = gwas_path
    
    if geno_path is None:
        target_train_data_path = '../data/'+model_flag+'/'+fname_root+'_PRS_train_'+str(itr)
        target_test_data_path = '../data/'+model_flag+'/'+fname_root+'_PRS_test_'+str(itr)
    else:
        target_train_data_path = geno_path.split(' ')[0]   
        target_test_data_path = geno_path.split(' ')[1]      
    
    if pheno_path is None:
        pheno_train_data_path = '../data/'+model_flag+'/'+fname_root+'_pheno_out_train_'+str(itr)+'.txt'
        pheno_test_data_path = '../data/'+model_flag+'/'+fname_root+'_pheno_out_test_'+str(itr)+'.txt'
    else:
        pheno_train_data_path = pheno_path.split(' ')[0]   
        pheno_test_data_path = pheno_path.split(' ')[1]
    
    if covs_path is None:
        covs_train_data_path = '../data/'+model_flag+'/'+fname_root+'_PRS_train_PCA_out_'+str(itr)+'_singularVectors.txt'
        covs_test_data_path = '../data/'+model_flag+'/'+fname_root+'_PRS_test_PCA_out_'+str(itr)+'_singularVectors.txt'
    else:
        covs_train_data_path = covs_path.split(' ')[0]   
        covs_test_data_path = covs_path.split(' ')[1]

    ### Train
    prsice_train_cmd = 'Rscript ~/PRSice/PRSice.R --prsice ~/PRSice/PRSice_linux --base '+base_data_path+' --target '+target_train_data_path+' --binary-target '+str(binary_target)+' --pheno '+pheno_train_data_path+' --stat BETA --beta --cov '+covs_train_data_path+' --ignore-fid --keep-ambig --no-clump --out ../data/'+model_flag+'/'+fname_root+'_PRS_prsice_train_'+str(itr)
    ### Test
    prsice_test_cmd = 'Rscript ~/PRSice/PRSice.R --prsice ~/PRSice/PRSice_linux --base '+base_data_path+' --target '+target_test_data_path+' --binary-target '+str(binary_target)+' --pheno '+pheno_test_data_path+' --stat BETA --beta --cov '+covs_test_data_path+' --ignore-fid --keep-ambig --no-clump --out ../data/'+model_flag+'/'+fname_root+'_PRS_prsice_test_'+str(itr)

    # Shell calls
    subprocess.call(prsice_train_cmd, shell=True)
    subprocess.call(prsice_test_cmd, shell=True)
    
