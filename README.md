# FairPRS
A fairness framework for Polygenic Risk Scores

FairPRS proposes an Invariant Risk Minimization (IRM) approach for estimating fair PRS or debiasing pre-computed ones.
FairPRS offers an entire pipeline from genetic data to trait prediction. It has three possible access points for input: genotypes, genotypes with summary statistics, or a pre-computed PRS.

### Input output parameters
  
If starting at the genotype access point of the pipeline, the required inputs are 3 Plink files (BED,BIM,FAM) containing the genotype data for GWAS, FairPRS training and testing, respectively.
Moreover, to compute the PRS the following files are required per PRS computation.
- GWAS - summary statics for betas extraction. The same file is used for both PRS computations.
- Genotype - Plink files (BED,BIM,FAM) for each PRS train and test.
- Phenotype - It can be a stand alone file for each PRS computation or if not provided the phenotype will be extracted from the FAM file. The file should be a .txt file formated with 3 columns FID,IID,PHENO.
- Covariates - One file per PRS containing all the covaraites to be included. Formated with the first column being IID and as many columns as covariates.

If started at access point 1, there is no need to add more files as all required files are generated in step 1.

For access point 3 (FairPRS), one file containing the following data is required:
- EID - First column contains the sample IDs.
- PRS - Second column should contain all the PRS scores.
- Population IDs - Third column should have the phenotype.
- Phenotype - Fourth column should contain the phenotype (Please make sure this column's header is 'pheno')
- Covariates - The remaining columns should contain the covariates to include (e.g Age, Sex, PCs). Based on the covariates flag the number of these to be included can be selected.

Similarly to the previous access point, if started at access point 2, all required files are generated in step 1 except for the population IDs (which are provided as part of the demo data).

### Flags
- `-g --GWAS_PC_calc` -> Access point 1 of the pipeline. GWAS and PCA computation. Enter (1 - yes, 0 no) to run or skip this segment.
- `-p --PRS_calc` Access -> point 2 of the pipeline. PRS computation. Enter (1 - yes, 0 no) to run or skip this segment.
- `-fair  --fairPRS_calc` -> Access point 3 of the pipeline. Run FairPRS model training/tuning/evaluation. Enter (1 - yes, 0 no) to run or skip this segment.
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



### Demo run / call
#### Compute GWAS, PCs and train/tune model
```
python fairPRS.py -g 0 -p 1 -fair 0 --random_seed 42 -iters 1 -tun 1 -plot_inp_dist 1 -plot_out_dist 1 -envs 3 --trait 1 --model PSD -fi PSD_10_20_70_continuous_0.05 -num_pcs 8 -num_cov 0
```

### Repo structure
