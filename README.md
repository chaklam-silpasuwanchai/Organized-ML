# Organizing Machine Learning Projects
The repository aims to teach my students how to best organize your ML code for efficiency and to facilitate experiments.  This example is mostly for PyTorch but can be adjusted.  It contains backbone folders and some sample files for illustration.

table of contents
    [## Key idea]

## Key idea

When we build our project, we don't want to use a lot of our memory.  Basically, we want to run as mindlessly as possible.  We don't want to remember which models we just run, how many epochs, etc.   A

Also, we want to repeat ourself as little as possible.  When we experiment, we want to change as little code as possible.    Everything should be as well-isolated.  

## Sample structure

Here is a sample project structure you can start with.

    +-- data                    
    |   +-- raw                     #raw dataset you get from internet (non-modified)
    |       +-- subj1.csv
    |   +-- processed               #preprocessed dataset; ready to be used
    |       +-- subj1.npy       
    +-- src                         #contains .py files
    |   +-- _preprocess.py          #for preprocessing and constructing the dataset, e.g., subj1.npy
    |   +-- _build_feature.py       #for generating features
    |   +-- _train.py               #for training; also supports evaluation
    |   +-- _experiment.py          #for experiment
    |   +-- _utils.py               #useful utilities (used across projects)
    +-- configs                     #contain configurations for preprocessing and networks
    |   +-- example-config1.yaml
    |   +-- example-config2.yaml
    +-- models                      #for saving models
    |   +-- sample_model1.pkl
    +-- notebooks
    |   +-- sample_notebook.ipynb   #for exploring and plotting; can save figures to figures folder
    +-- networks                    #containing networks that we want to compare; use it with train.py
    |   +-- sample_network1.py  
    |   +-- sample_network2.py
    +-- scripts
    |   +-- run.sh                  #for running multiple experiments; this is the main file user will run...
    +-- figures
    +-- outputs
    |   +-- example-config1.txt     #output results for experiments
    requirements.txt
    .gitignore

We used underscore `_` to say that these are files that are not supposed to be used by users to run as interface.  

The main file that the user will interact is `run.sh`.  The `run.sh` will execute `experiment.py` which will in turn execute multiple `train.py` with different configurations.  `*-config.yaml` will define which network architecture, hyperparameters, and preprocessed file to use. 

Optionally, you can also have these folders:

    +-- docker                  #contains docker related files, e.g Dockerfile
    |   +-- Dockerfile
    +-- api
    |   +-- api.py              #expose APIs for production
    +-- logs
    |   +-- sample-log

 ## Case Study: MNIST

 For simplicity, we just use the MNIST as case study.

 We download the data, `mnist_test.csv` and `mnist_train.csv` and put in inside `data/raw` directory.

    pip install pyyaml