# SpectAcle

This repository reports the code and the benchmarks for the paper "SpectAcle: Fault Localisation of AI-Enabled CPS by Exploiting Sequences of DNN Controller Inferences".

## System Requirements

- Operating system: Linux / MacOS / Windows
- Matlab (Simulink/Stateflow) version: >= 2020a (Matlab license needed)
- Python version: >= 3.3
- MATLAB toolboxes dependency: 
  1. [Simulink](https://www.mathworks.com/products/simulink.html)
  2. [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)

## Installation

- Install [Breach](https://github.com/decyphir/breach)
  1. start matlab, set up a C/C++ compiler using the command `mex -setup`. (Refer to [here](https://www.mathworks.com/help/matlab/matlabexternal/changing-default-compiler.html) for more details.)
  2. navigate to `breach/` in Matlab command line, and run `InstallBreach`
- Install Git Large File Storage
  1. for linux:
     ```
     curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
     sudo apt-get install git-lfs
     git lfs install
     ```
  2. for mac:
     ```
     brew install git-lfs
     git lfs install
     ```
- Retrieve the dataset stored in the form of a large file by the following command `git lfs pull`

 ## Preparation

 ### NN Controller Training
 
  - Navigate to the directory `test/`. Run the command `python train.py config/[benchmark]/train`.
  - Now the executable scripts have been generated under the directory `test/scripts/`. 
  - Navigate to the root directory `Spectacle/` and run the command `make`. The automatically generated NN controller files will be stored under the root directory.

 ### Construction of Benchmarks

 - Navigate to the directory `test/`. Run the command `python mutation.py config/[benchmark]/mutation[number]`.
 - Now the executable scripts have been generated under the directory `test/scripts/`. 
 - Navigate to the root directory `Spectacle/` and run the command `make`. The neccessary simulation logs of mutants for fault localisation will be stored under the root directory, including the diagnostic information of current test suite `cur_diagInfo_suite`, the safety rate of current mutant on the given test suite `cur_safety_rate`, the Boolean state of current mutant for each test case `cur_state_set` (either 1 (satisfied) or 0 (violated)), and the mutator's information `mut_info`. 

## Usage

 To reproduce the experimental results, users should follow the steps below:
 
  - The user-specified configuration files are stored under the directory `test/config/`. Replace the paths of `Spectacle` in user-specified file under the line `addpath 1` with their own path. Users can also specify other configurations, such as topk and tops.
  - Users need to edit the executable scripts permission using the command `chmod -R 777 *`.
  - The corresponding log will be stored under directory `output/`.
    
 ### Fault Localisation

 - Navigate to the directory `test/`. Run the command `python spectacle.py config/[benchmark]/fl[number]`.
 - Now the executable scripts have been generated under the directory `test/scripts/`. 
 - Navigate to the root directory `Spectacle/` and run the command `make`. The fault localisation results will be stored under the root directory, including the mutator's information `mut_info`, different hyperparameter combinations for fault localisation `hyper_config`, the execution spectra under different hyperparameter combinations `cur_hyper_exe_spectrum`, the suspicious weights under different hyperparameter combinations `cur_hyper_all_sps_weight`, and the suspicious scores of all weights under different hyperparameter combinations cur_hyper_all_sps_score.  

## Reproduction of Experimental Results

### RQ1: Does SpectAcle effectively localise the faulty DNN weights in AI-enabled CPSs?

Based on the simulations logs and fault localisation results, all the tables and figures can be obtained through the following scripts:
 - DR Plots: RQ1_DR_plot.m 
 - $AUC_{DR}$: RQ1_AUC_DR.m
 - $AUC_{FDR}$: RQ1_AUC_FDR.m

### RQ2: How does the selection of hyperparameter k influence the effectiveness of SpectAcle?

 - Statistical Comparison in terms of $AUC_{DR}$: RQ1_Cohens_d_DR_hyperparm.m
 - Statistical Comparison in terms of $AUC_{FDR}$: RQ1_Cohens_d_FDR_hyperparm.m

### RQ3: How do $SpectAcle_{w}$ and $SpectAcle_{uw}$ compare with each other? 

 - Comparison between $SpectAcle_{uw}$ and $SpectAcle_{w}$: RQ3.m

### RQ4: How does the selection of the suspiciousness metric SusMetr influence the effectiveness of SpectAcle?

 - Statistical Comparison in terms of $AUC_{DR}$: RQ1_Cohens_d_DR_metric.m
 - Statistical Comparison in terms of $AUC_{FDR}$: RQ1_Cohens_d_FDR_metric.m
  
### RQ5: Are the weights identified by SpectAcle useful for improving the DNN controller performance?

 - Navigate to the directory `test/`. Run the command `python contrRep4SpectAcle.py config/[benchmark]/repair[number]`.
 - Now the executable scripts have been generated under the directory `test/scripts/`. 
 - Navigate to the root directory `Spectacle/` and run the command `make`. The repair results will be stored under the root directory.

### RQ6: What is the computational cost of SpectAcle?

 - The time cost of SpectAcle: RQ6.m
