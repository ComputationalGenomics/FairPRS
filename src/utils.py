import os, argparse, time, subprocess, string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from prs_dataset import PRS
import pickle

# Plotting PRS distributions by pop
def plot_prs_dist(prs_fname = None ,popids_fname= None, PRS_df = None, popids_df = None, fname_mod = None, path = '../data/', mod_name=None, simrel=None):
    sns.set_style('darkgrid')
    if popids_df is None:
        pop_ids = pd.read_csv(str(path+'/'+popids_fname[0]+'pop_ids_'+popids_fname[1]+'.txt'),header = None)
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
        plt.savefig(str('../results/'+mod_name+'/plots/'+popids_fname[0]+popids_fname[1]+'.png'))
    else:
        plt.savefig(str('../results/'+mod_name+'/plots/'+fname_mod+'.png'))
    plt.clf()
    
def plot_pheno_dist(prs_fname = None ,popids_fname= None, PRS_df = None, popids_df = None, fname_mod = None, path = '../data/', mod_name=None, simrel=None):
    sns.set_style('darkgrid')
    if popids_df is None:
        pop_ids = pd.read_csv(str(path+'/'+popids_fname[0]+'pop_ids_'+popids_fname[1]+'.txt'),header = None)
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
        plt.savefig(str('../results/'+mod_name+'/plots/'+popids_fname[0]+popids_fname[1]+'.png'))
    else:
        plt.savefig(str('../results/'+mod_name+'/plots/'+fname_mod+'.png'))
    plt.clf()

def plot_out(results_dict_irm, plot_fname, model_flag, train_test, simrel):
    # Plot predicted PRS distributions per population
    ancs = pd.DataFrame(results_dict_irm['data'][str('ancs_'+train_test)])
    print(ancs)
    uniques = np.unique(ancs.iloc[:,0], return_counts = True)[0]
    ancs_mapping_inv = dict(zip(np.arange(len(uniques)),uniques))
    ancs.iloc[:,0] = ancs.iloc[:,0].map(ancs_mapping_inv)
    PRS_pred = pd.DataFrame(results_dict_irm['data'][str('PRS_pred_'+train_test)])
    PRS_pred.columns = ['PRS']
    plot_prs_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = str(plot_fname[0]+'_PRS_predicted'+plot_fname[1]), mod_name=model_flag, simrel=simrel)

    # Plot predicted Phenotype distributions per population
    ancs = pd.DataFrame(results_dict_irm['data'][str('ancs_'+train_test)])
    uniques = np.unique(ancs.iloc[:,0], return_counts = True)[0]
    ancs_mapping_inv = dict(zip(np.arange(len(uniques)),uniques))
    ancs.iloc[:,0] = ancs.iloc[:,0].map(ancs_mapping_inv)
    PRS_pred = pd.DataFrame(results_dict_irm['data'][str('Pheno_og_'+train_test)])
    PRS_pred.columns = ['Phenotype']
    plot_pheno_dist(popids_df=ancs,PRS_df = PRS_pred, fname_mod = str(plot_fname[0]+'_Phenotype_predicted'+plot_fname[1]), mod_name=model_flag, simrel=simrel)

def results_summary(results_dict, mod_name, model_flag, fname_root_out, itr, saveres):
    # Load results from all iterations for summary results                 
    temp = []
    for k in results_dict.keys():
        temp.append(pd.DataFrame.from_dict(results_dict[k]['metrics']))
        
    metrics_df = pd.concat(temp)
    metrics_df_out =  pd.DataFrame(metrics_df.loc[:,['loss_train','loss_test','R2_og_test','R2_pred_test','r2_prs_train','r2_prs_test','r2_pheno_train','r2_pheno_test' ]].mean()).T
    print(f'Average {mod_name} model results: train MSE {round(metrics_df_out["loss_train"][0],5)}, test MSE {round(metrics_df_out["loss_test"][0],5)}, R2 original PRS {round(metrics_df_out["R2_og_test"][0],5)}, R2 predicted PRS {round(metrics_df_out["R2_pred_test"][0],5)}, r2 prs {round(metrics_df_out["r2_prs_test"][0],5)},r2 pheno {round(metrics_df_out["r2_pheno_test"][0],5)}')

    if saveres:
        with open(str('../results/'+model_flag+'/'+mod_name+'_'+fname_root_out+'_'+str(itr)+'_iters_results_dictionary.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)
    return metrics_df_out

def plot_ins(X, fname_root_out, itr, model_flag, train_test, ancs_mapping_inv):
    # Plot PRS dist for real data all 
    popids_df = pd.DataFrame(X[:,2])
    popids_df.iloc[:,0] = popids_df.iloc[:,0].map(ancs_mapping_inv)
    PRS_df = pd.DataFrame(X[:,1])
    PRS_df.columns = ['PRS']
    plot_fname_train = fname_root_out+'_PRS_'+train_test+'_'+str(itr)
    plot_prs_dist(popids_df=popids_df,PRS_df = PRS_df, fname_mod = plot_fname_train, mod_name=model_flag, simrel=0)
    print('Saved KDE plots in results directory')

def data_prep(real_data_df, val_size, test_size, num_pcs, num_covs, num_envs, rnd_state, plot_input_dists):
    # From single dataframe train/val/test pytorch datasets
    # if real PRS data with only one input file -> read and split into train/val/test
    # Real data should be a data frame with eids, prs, pheno, ancs, :covariates
    uniques = np.unique(real_data_df['ancs'], return_counts = True)[0]
    ancs_mapping = dict(zip(uniques,np.arange(len(uniques))))
    ancs_mapping_inv = dict(zip(np.arange(len(uniques)),uniques))
    real_data_df['ancs'] = real_data_df['ancs'].map(ancs_mapping)

    X_train, X_test, y_train, y_test = train_test_split(real_data_df.loc[:,real_data_df.columns != 'pheno'], real_data_df['pheno'], test_size=test_size, random_state=rnd_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=rnd_state)

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
    
    pcs_train = X_train[:,3:(num_covs + num_pcs + 3)]
    pcs_val = X_val[:,3:(num_covs + num_pcs + 3)]
    pcs_test = X_test[:,3:(num_covs + num_pcs + 3)]

    # Split train into enviornments
    ids_train_chunks = np.array_split(X_train[:,0],num_envs)
    prs_train_chunks = np.array_split(X_train[:,1],num_envs)
    pop_ids_train_chunks = np.array_split(pop_ids_train,num_envs)
    y_train_chunks = np.array_split(y_train,num_envs)
    pcs_train_chunks = np.array_split(pcs_train,num_envs)

    train_datasets = []
    for i in range(num_envs):
        train_datasets.append(PRS(ids_train_chunks[i], prs_train_chunks[i],y_train_chunks[i],pcs_train_chunks[i],pop_ids_train_chunks[i]))
    
    all_train_data = PRS(X_train[:,0], X_train[:,1],y_train, pcs_train, pop_ids_train)
    val_data = PRS(X_val[:,0], X_val[:,1], y_val, pcs_val, pop_ids_val)
    test_data = PRS(X_test[:,0], X_test[:,1], y_test, pcs_test, pop_ids_test)
    
    if plot_input_dists:
        return all_train_data, train_datasets, val_data, test_data, X_train, X_test, ancs_mapping_inv
    else:
        return all_train_data, train_datasets, val_data, test_data

def data_prep_simdata(model_flag, fname_root, itr, num_envs):
    # Load PRS and pheno
    # Load train and test PRS
    plink_prs_train = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_prsice_train_'+str(itr)+'.best'), sep = '\s+')
    plink_prs_test = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_prsice_test_'+str(itr)+'.best'), sep = '\s+')
    prs_train = plink_prs_train.PRS.values
    prs_test = plink_prs_test.PRS.values

    # add column with ancestry name
    pop_ids_train = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_train_pop_ids_'+str(itr)+'.txt'),header = None)
    pop_ids_test = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_test_pop_ids_'+str(itr)+'.txt'),header = None)
    pop_ids_train = pop_ids_train.values
    pop_ids_test = pop_ids_test.values

    # Load pheno and keep as y
    pheno_train  = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_train_'+str(itr)+'.fam'), sep = '\t', header = None)
    pheno_test  = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_test_'+str(itr)+'.fam'), sep = '\t', header = None)
    y_train = pheno_train.iloc[:,5].values
    y_train = np.expand_dims(y_train,1)
    y_test = pheno_test.iloc[:,5].values
    y_test = np.expand_dims(y_test,1)

    # Load pcs 
    pcs_train = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_train_PCA_out_'+str(itr)+'_singularVectors.txt'), sep = '\t')
    pcs_train = pcs_train.values[:,1:]

    pcs_test = pd.read_csv(str('../data/'+model_flag+'/'+fname_root+'_PRS_test_PCA_out_'+str(itr)+'_singularVectors.txt'), sep = '\t')
    pcs_test = pcs_test.values[:,1:]

    # Split train into enviornments
    ids_train_chunks = np.array_split(np.arange(len(prs_train)),num_envs)
    prs_train_chunks = np.array_split(prs_train,int(num_envs))
    pop_ids_train_chunks = np.array_split(pop_ids_train,int(num_envs))
    pcs_train_chunks = np.array_split(pcs_train,int(num_envs))
    y_train_chunks = np.array_split(y_train,int(num_envs)) 
    
    if itr%10 == 0:
        print('Train and Test data successfully read in')
    
    ## Create dataloaders - train_n and test
    # Load and split into train test sets 
    train_datasets = []
    for i in range(int(num_envs)):
        train_datasets.append(PRS(ids_train_chunks[i],prs_train_chunks[i],y_train_chunks[i],pcs_train_chunks[i],pop_ids_train_chunks[i]))

    all_train_data = PRS(np.arange(len(prs_train)),prs_train,y_train,pcs_train,pop_ids_train)
    test_data = PRS(np.arange(len(prs_train)),prs_test, y_test, pcs_test, pop_ids_test)

    return all_train_data, train_datasets, test_data

def plot_ins_simdata(fname_root, itr, model_flag):
    # Set population id file names for loading
    popids_train_fname = [fname_root+'_PRS_train_',str(itr)]
    popids_test_fname = [fname_root+'_PRS_test_',str(itr)]
    # Plot PRS train and test distributions by ancestry (population id) group
    plot_prs_dist(str(fname_root+'_PRS_prsice_train_'+str(itr)+'.best'),popids_train_fname, path=str('../data/'+model_flag), mod_name=model_flag, simrel=1)
    plot_prs_dist(str(fname_root+'_PRS_prsice_test_'+str(itr)+'.best'),popids_test_fname, path=str('../data/'+model_flag), mod_name=model_flag, simrel=1)