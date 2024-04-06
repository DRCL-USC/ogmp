# OGMP

Codebase accompanying the paper [OGMP: Oracle Guided Multimodal Policies for Agile and Versatile Robot Control](https://arxiv.org/abs/2403.04205)


<p align="center">
   <img width="400" height="226" src="media/parkour.gif">
   <img width="400" height="226" src="media/dive_all4.gif">
</p>

## Installation

To install the dependencies, run:
    
    pip3 install -r requirements.txt

Install old version of torch (new may work but not tested)
    
    pip3 install torch==1.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

tested in python 3.8.10

## Usage

To train the policy, run:

    python train.py --exp_file exp_confs/dive_train.yaml

To test the policy, run:
    
    python test.py --exp_file exp_confs/dive_test.yaml

## High-level overview

* algos: contains the custom ppo implementation from [link](https://github.com/osudrl/RSS-2020-learning-memory-based-control)
* analysis: contains the code for analysis in the paper
* nn: contains the custom torch neural network (FF and LSTM), policy, critic implementation.
* logs: contains the training logs.
* dtsd: contains the environments
* exp_confs: contains the experiment configuration files for training and testing.
* train.py: file to train the policies.
* test.py: file to test the policies.

## To Do 

- [ ] add the training configuration to recreate the results in the paper
- [ ] train with new environments with the best config
- [ ] test with the best config
- [ ] analysis codes
