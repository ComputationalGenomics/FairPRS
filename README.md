# FairPRS
A fairness framework for Polygenic Risk Scores

FairPRS offers an entire pipeline from genetic data to trait prediction. It has three possible access points for input: genotypes, genotypes with summary statistics, or a pre-computed PRS.

FairPRS proposes an Invariant Risk Minimization (IRM) approach for estimating fair PRS or debiasing pre-computed ones.


### Input output parameters

If starting at the genotype access point of the pipeline, the required inputs are 3 Plink files (BED,BIM,FAM) containing the genotype data for GWAS, FairPRS training and testing, respectively. Moreover, 3 .txt files containing the population ids for all the samples is required. 

Summary stats


### Demo run / call
#### Compute GWAS, PCs and train/tune model
>> python fairPRS.py -sim_data 1 --prop 10,20,70 -szs 10000,1000,400 -snps 100000 -causal 0.05 --random_seed 42 -iters 10 -tun 0 -plot_inp_dist 1 -envs 3 -model_train 0 --trait 1 --model PSD -num_pop 3 -num_pcs 8 -r 1
#### Load real data, train/tune model
>> python fairPRS.py -sim_data 0 --random_seed 42 -iters 10 -tun 1 -plot_inp_dist 0 -envs 3 -model_train 1 --trait 1 --model PSD -num_pop 3 -train_fname PRS_prsice_train_PSD_10_20_70_continuous_0.05_0.best -test_fname PRS_prsice_test_PSD_10_20_70_continuous_0.05_0.best



Brief explanation of pipeline and repo structure
