import subprocess
import pandas as pd
import numpy as np

def summary_stats(model_flag, fname_root, num_pcs, itr, man_plot):
    # Set up common name root for all files
    # if args.prefix is not None:
    #     fname_root = args.prefix
    # else:
    #     fname_root = args.geno_load_path.split('/')[-1]

    ### Compute GWAS ###
    if itr%10 == 0:
        print('>>>>>>>>>>>>>>>>>>>>>> Computing GWAS <<<<<<<<<<<<<<<<<<<<<<')
    # Compute PCS / covariates
    tera_pca_train_cmd = '~/TeraPCA/TeraPCA.exe -bfile ../data/'+model_flag+'/'+fname_root+'_train_'+str(itr)+' -nsv '+num_pcs+' -filewrite 1 -prefix ../data/'+model_flag+'/'+fname_root+'_train_PCA_out_'+str(itr)
    tera_pca_prs_train_cmd = '~/TeraPCA/TeraPCA.exe -bfile ../data/'+model_flag+'/'+fname_root+'_PRS_train_'+str(itr)+' -nsv '+num_pcs+' -filewrite 1 -prefix ../data/'+model_flag+'/'+fname_root+'_PRS_train_PCA_out_'+str(itr)
    tera_pca_prs_test_cmd = '~/TeraPCA/TeraPCA.exe -bfile ../data/'+model_flag+'/'+fname_root+'_PRS_test_'+str(itr)+' -nsv '+num_pcs+' -filewrite 1 -prefix ../data/'+model_flag+'/'+fname_root+'_PRS_test_PCA_out_'+str(itr)
    subprocess.call(tera_pca_train_cmd, shell=True)
    subprocess.call(tera_pca_prs_train_cmd, shell=True)
    subprocess.call(tera_pca_prs_test_cmd, shell=True)
    #Load covariates and adapt to plink format - covariate
    cov = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_train_PCA_out_'+str(itr)+'_singularVectors.txt'), sep='\t')
    cov.insert(1, 'IID', cov['FID'])
    cov_np = cov.to_numpy(dtype='str')
    cov_out = np.insert(cov_np,0,cov.columns.values, axis =0)
    np.savetxt(str('../data/'+model_flag+'/'+fname_root+'_PCA_out_covariates_'+str(itr)+'.cov'), cov_out, delimiter='\t', fmt='%s')

    # Run GWAS with plink GLM
    gwas_cmd = 'plink2 --bfile ../data/'+model_flag+'/'+fname_root+'_train_'+str(itr)+' --glm --out ../data/'+model_flag+'/'+fname_root+'_GLM_results_'+str(itr)+' --covar ../data/'+model_flag+'/'+fname_root+'_PCA_out_covariates_'+str(itr)+'.cov'
    subprocess.call(gwas_cmd, shell=True)

    # Update GLM (GWAS) headers for standard use on PRS software and Manhattan plotting library
    nam = '../data/'+model_flag+'/'+fname_root+'_GLM_results_'+str(itr)+'.PHENO1.glm.linear'
    GLM = pd.read_csv(nam, sep = '\s+')
    GLM.columns = ['CHR', 'BP', 'SNP', 'A2', 'ALT','A1', 'TEST','OBS_CT', 'BETA', 'SE', 'T_STAT', 'P']
    # Drop all duplicated SNPs / only leave values where TEST column = ADD (Band aid fix)
    GLM = GLM.loc[GLM['TEST'] =='ADD']
    # Filling Na values with column means
    GLM['BETA'] = GLM['BETA'].fillna(GLM['BETA'].mean())
    GLM['SE'] = GLM['SE'].fillna(GLM['SE'].mean())
    GLM['T_STAT'] = GLM['T_STAT'].fillna(GLM['T_STAT'].mean())
    GLM['P'] = GLM['P'].fillna(GLM['P'].mean())
    GLM.to_csv(str('../data/'+model_flag+'/'+fname_root+'_GLM_DF_'+str(itr)+'.txt'), sep='\t', index = False)

    if man_plot:
    # Plot GWAS - Manhattan plot 
        print('>>>>>>>>>>>>>>>>>> Plotting Manhattan plot for GWAS <<<<<<<<<<<<<<<<<<<<<<')

        ## Need to find alternative way to qmplot, takes too long to ouput image !!!
        man_plot_cmd = 'qmplot -I '+nam+' -T '+fname_root+' --dpi 300 -O '+'../results/'+fname_root
        subprocess.call(man_plot_cmd, shell=True)
        print('>>>> Saved on Results folder <<<<<')
