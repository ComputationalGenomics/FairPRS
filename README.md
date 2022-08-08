# FairPRS
A fairness framework for Polygenic Risk Scores

FairPRS proposes an Invariant Risk Minimization (IRM) approach for estimating fair PRS or debiasing pre-computed ones.
FairPRS offers an entire pipeline from genetic data to trait prediction. It has three possible access points for input: genotypes, genotypes with summary statistics, or a pre-computed PRS.

### Input - Output parameters
  
If starting at the genotype access point of the pipeline, the required inputs are:
- Plink files (BED,BIM,FAM) containing the genotype data for GWAS, FairPRS training and testing, respectively (total of 9).
- For example, geno_train.bed/bim/fam, geno_prs_train.bed/bim/fam, geno_prs_test.bed/bim/fam

Access point 1 (`--access geno`) will save files for the formated summary statistics, and covariates (age, sex, principal component (PC), etc.) files for each input in the data directory.


Moreover, to compute the PRS the following files are required per PRS computation.
- GWAS - summary statics for betas extraction. The same file is used for both PRS computations.
- Genotype - Plink files (BED,BIM,FAM) for each PRS train and test.
- Phenotype - It can be a stand alone file for each PRS computation or if not provided the phenotype will be extracted from the FAM file. The file should be a .txt file formated with 3 columns FID,IID,PHENO.
- Covariates - One file per PRS containing all the covaraites to be included. Formated with the first column being IID and as many columns as covariates.

If started at access point 1, there is no need to add more files as all required files are generated in step 1.

Access point 2 (`--access sumstats`) will save files containing PRS scores for each input in the data directory.
If `plot_inp_dist` flag activated, it will also save KDE plots for each input with the PRS distributions by population in the results directory.

For access point 3 (`--access prs`), one file containing the following data is required:
- EID - First column contains the sample IDs.
- PRS - Second column should contain all the PRS scores.
- Population IDs - Third column should have the phenotype.
- Phenotype - Fourth column should contain the phenotype (Please make sure this column's header is 'pheno')
- Covariates - The remaining columns should contain the covariates to include (e.g Age, Sex, PCs). Based on the covariates flag the number of these to be included can be selected.

Similarly to the previous access point, if started at access point 2, all required files are generated in step 1 except for the population IDs (which are provided as part of the demo data).

If `plot_out_dist` and/or `plot_out_dist` flags are activated, it will also save KDE plots for each input/output with the PRS distributions by population.

Access point 3 (`-fair 1`) will save files containing the summary of results over the iterations an a dictionary containing all the predicted and original PRS, and results, hyperparams per iteration. Results dictionary structure per iteration is as follows:

results_ret = {
- 'metrics':{'R2_og_train', 'R2_og_val', 'R2_og_test', 'R2_pred_train', 'R2_pred_val', 'R2_pred_test', 'r2_prs_train', 'r2_prs_val', 'r2_prs_test', 'r2_pheno_train', 'r2_pheno_val', 'r2_pheno_test', 'loss_train', 'loss_val', 'loss_test'},
- 'data':{ 'PRS_og_train', 'PRS_og_val', 'PRS_og_test', 'PRS_pred_train', 'PRS_pred_val', 'PRS_pred_test', 'Pheno_og_train', 'Pheno_og_val', 'Pheno_og_test', 'Pheno_pred_train', 'Pheno_pred_val', 'Pheno_pred_test', 'ancs_train', 'ancs_val', 'ancs_test', 'pcs_train', 'pcs_val', 'pcs_test', 'deciles_r2s_pred_train',  'deciles_r2s_pred_val', 'deciles_r2s_pred_test', 'deciles_r2s_og_train', 'deciles_r2s_og_val', 'deciles_r2s_og_test'},
- 'hyperparams':{
          'lr', 'pen_mult', 'units'}
      
  }


### Flags
-  `-a --access` -> Access point flag. Takes comma separated inputs {geno, sumstat, prs}. E.g. `--access sumstat,prs`. Access points: (geno) - GWAS and PCA computation, (sumstat) - PRS computation, (prs) - Run FairPRS model training/tuning/evaluation.
- `-d  --dir` -> Path to the real PRS dataset when starting at access point 3.
- `-plot_inp_dist  --plot_input_distributions` -> Enter (1 - yes, 0 no) to plot input PRS distributions.
- `-plot_out_dist  --plot_output_distributions` -> Enter (1 - yes, 0 no) to plot output PRS and phenotype distributions (after FairPRS).
- `-man_plot  --manhattan_plot` -> Enter (1 - yes, 0 no) to plot Manhattan plot of GWAS at access point 1.
- `-fi  --fname_prefix_root` -> Enter the prefix of your files in case of starting at access point 1.
- `-fo  --fname_out_root` -> Enter the prefix for all output files.
- `-gpu  --gpu_bol` ->  Enter (1 - yes, 0 no) to use GPU for FairPRS (default = 0 -> CPU)
- `-num_pcs  --num_pcs` -> Enter the number of PCs to calculate in access point 1 and or to use in trainig for FairPRS.
- `-num_covs  --num_covs` -> Enter the number of covariates to use in FairPRS additional to the PCs.
- `-pheno_fname  --pheno_fname` -> Enter paths to pheno train and test files separated by a space for PRS computation at access point 2.
- `-gwas_fname  --gwas_fname` -> Enter path to summary statistics file for PRS computation at access point 2.
- `-geno_fname   --geno_fname` -> Enter paths to genotype train and test files separated by a space for PRS computation at access point 2.
- `-covs_fname --covs_fname` -> Enter paths to covariates for train and test separated by a space for PRS computation at access point 2.
- `-iters --iterations` -> Enter the number of iterations to run the FairPRS pipeline.
- `-envs --num_pop_envs` -> Enter the number of populations present in dataset. 
- `-m --model` -> Enter the name of the genetic model. This will be used to create the subdirectory in /data and /results for the experiment.
- `-tf --trait 1` -> Enter (1 - continuous , 0 binary)



### Demo run / call
#### Compute PCs and train/tune model with sample data
```
python fairPRS_wrapper.py --access geno,sumstats,prs --random_seed 42 -iters 1 -tun 1 -plot_inp_dist 1 -plot_out_dist 1 -envs 3 --trait 1 --model PSD -num_pop 6 -fi PSD_10_20_70_continuous_0.05 -num_pcs 8 -num_cov 0
```
Note: Option `--access geno` is disabled as the genotype files for GWAS are too large to be hosted in GitHub.
After cloning the repo `cd src` and run the comand above. It will perform the baseline PRS calculation, train, tune and evaluate FairPRS using the available sample data.
#### Compute train/tune model with real data
```
python fairPRS_wrapper.py --access prs -iters 10 -plot_inp_dist 1 -plot_out_dist 1 -envs 6 --trait 1 --model PSD -fo UKB_height -d ../data/UKB_height/UKB_height_df.csv -num_pcs 8 -num_cov 0
```
Note: UKB data is not included in this repository, command just for ilustration purposes.

### Repo structure
Two main folders, src and data. src contains the FairPRS code, while data contains all the sample data described above.

### Contact
Diego Machado Reyes (machad at rpi dot edu) 

Aritra Bose (a dot bose at ibm dot com)
