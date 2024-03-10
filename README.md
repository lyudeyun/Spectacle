# SpectAcle

This repository reports the source code and the twelve benchmarks for the paper "SpectAcle: Fault Localisation of AI-Enabled CPS by Exploiting Sequences of DNN Controller Inferences".

## System requirement

- Operating system: Linux or MacOS;
- Matlab (Simulink/Stateflow) version: >= 2021a. (Matlab license needed)
- Python version: >= 3.3
- MATLAB toolboxes dependency: 
  1. [Simulink](https://www.mathworks.com/products/simulink.html)
  2. [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) 

## Installation

- Install [Breach](https://github.com/decyphir/breach)
  1. start matlab, set up a C/C++ compiler using the command `mex -setup`. (Refer to [here](https://www.mathworks.com/help/matlab/matlabexternal/changing-default-compiler.html) for more details.)
  2. navigate to `breach/` in Matlab commandline, and run `InstallBreach`
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

 ## Usage

 To reproduce the experimental results, users should follow the steps below:
 
  - The user-specified configuration files are stored under the directory `test/config/`. Replace the paths of `Spectacle` in user-specified file under the line `addpath 1` with their own path. Users can also specify other configurations, such as topk and tops.
  - Users need to edit the executable scripts permission using the command `chmod -R 777 *`.
  - The corresponding log will be stored under directory `output/`.
    
 ### Mutation for Validation of Spectacle

 - Navigate to the directory `test/`. Run the command `python mutation.py config/[benchmark]/mutation[number]`.
 - Now the executable scripts have been generated under the directory `test/scripts/`. 
 - Navigate to the root directory `Spectacle/` and run the command `make`. The automatically generated files will be stored under the root directory, including the neccessary information of mutants for fault localisation.

 
 ### Fault Localisation

 - Navigate to the directory `test/`. Run the command `python spectacle.py config/[benchmark]/fl[number]`.
 - Now the executable scripts have been generated under the directory `test/scripts/`. 
 - Navigate to the root directory `Spectacle/` and run the command `make`. The automatically generated files will be stored under the root directory, including the fault localisation informations.
 
 ### NN Controller Training
 
  - Navigate to the directory `test/`. Run the command `python train.py config/[benchmark]/train`.
  - Now the executable scripts have been generated under the directory `test/scripts/`. 
  - Navigate to the root directory `Spectacle/` and run the command `make`. The automatically generated NN controller files will be stored under the root directory.
