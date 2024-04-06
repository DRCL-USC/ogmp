# OGMP

Codebase accompanying the paper [OGMP: Oracle Guided Multimodal Policies for Agile and Versatile Robot Control](https://arxiv.org/abs/2403.04205)


<p align="center">
   <img width="200" height="180" src="media/parkour.gif">
   <img width="200" height="180" src="media/dive_all.gif">
</p>

## Installation

    pip install -r requirements.txt

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
