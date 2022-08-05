import os, time
import hyperparam_tuning_pcs as ht
from utils import plot_ins, plot_out, data_prep, results_summary

def fairprs(real_data_df, iters = 1, plot_input_dists = True, plot_output_dists = True, test_size = 0.10 , val_size = 0.22, rnd_state = 42, gpu = True, num_pcs = 10, num_envs = 6, model_flag = None, trait_flag = 1, num_covs = 0, fname_root_out = None):
    begin_time = time.time()
    results_erm_tun = {}
    results_irm_tun = {}
    if not os.path.isdir(str('../results/'+model_flag)):
        os.system(str('mkdir ../results/'+model_flag))
    if not os.path.isdir(str('../results/'+model_flag+'/plots')):
        os.system(str('mkdir ../results/'+model_flag+'/plots'))

    for itr in range(iters):
        if itr%10 == 0:
            print('>>>>>>>>>>>>>>>>>>>>>> Loading and processing data <<<<<<<<<<<<<<<<<<<<<<')
        # Prepare datasets for pytorch loading
        if plot_input_dists:
            all_train_data, train_datasets, val_data, test_data, X_train, X_test = data_prep(real_data_df, val_size, test_size, num_pcs, num_covs, num_envs, rnd_state, plot_input_dists)
        else:
            all_train_data, train_datasets, val_data, test_data = data_prep(real_data_df, val_size, test_size, num_pcs, num_covs, num_envs, rnd_state, plot_input_dists)
        
        # Plot input PRS distributions
        if plot_input_dists:
            if itr%10 == 0:
                print('>>>>>>>>>>>>>>>>>>>>>> Plotting PRS distributions <<<<<<<<<<<<<<<<<<<<<<')
            plot_ins(X_train, fname_root_out, itr, model_flag, 'train')
            plot_ins(X_test, fname_root_out, itr, model_flag, 'test')

        ### Run FairPRS autotunning ###
        print('>>>>>>>>>>>>>>>>>>>>>> Autotuning and Evaluating FairPRS <<<<<<<<<<<<<<<<<<<<<<')
        if gpu:
            device = 'gpu'
            gpu_num = 1
        else:
            device = 'cpu'
            gpu_num = 0

        # Run training, hyperparam search and testing for ERM and IRM
        results_dict_erm = ht.tuner(all_train_data, test_data, irm=False, trait = int(trait_flag), num_pcs = (num_covs+num_pcs), device = device, gpus_per_trial=gpu_num)

        results_dict_irm = ht.tuner(train_datasets, test_data, all_train_data = all_train_data, irm=True, num_envs = num_envs, trait = int(trait_flag), num_pcs = (num_covs+num_pcs), device = device, gpus_per_trial=gpu_num)

        results_erm_tun[itr] = results_dict_erm
        results_irm_tun[itr] = results_dict_irm

        # Plot output PRS distributions
        if bool(int(plot_output_dists)):
            plot_fname = fname_root_out+'_PRS_test_predicted_'+str(itr)
            plot_out(results_dict_irm, plot_fname, model_flag, 'test')
            plot_fname = fname_root_out+'_PRS_train_predicted_'+str(itr)
            plot_out(results_dict_irm, plot_fname, model_flag, 'train')
            plot_fname = fname_root_out+'_Phenotype_test_'+str(itr)
            plot_out(results_dict_irm, plot_fname, model_flag, 'test')
            plot_fname = fname_root_out+'_Phenotype_test_predicted_'+str(itr)
            plot_out(results_dict_irm, plot_fname, model_flag, 'train')
            
    # Results summary and return full results dictionary 
    results_df_erm = results_summary(results_dict_erm, 'ERM')
    results_df_irm = results_summary(results_dict_irm, 'IRM')
        
    if (itr+1)%10 == 0:
        print(f'{itr} iterations completed.')
        print('='*40)
        
    print("--- Total run time in %s seconds ---" % (time.time() - begin_time))
    return results_erm_tun,results_irm_tun , results_df_erm, results_df_irm