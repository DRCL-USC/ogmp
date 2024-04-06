# CPPTF

## Installation

Since this is a codebase under development, the best practice is to try the relavent files and install the dependencies as you go along. At a high level we need generic modules like numpy, matplotlib, torch, gym, dm_control and/or mujoco, etc.

## Usage

The three major agenda to this codebase and the correspondin documentation are: 
    
1. To train rl polcicies for control. read: [./documentation/training.md](./documentation/training.md).

2. To test those trained rl policices in simulation. read: [./documentation/testing.md](./documentation/testing.md).

3. Do some analysis and tests to understand the policy and quantify its performance. read: [./documentation/analysis.md](./documentation/analysis.md).

## High-level overview

* algos: contains the custom ppo implementation.
* nn: contains the custom torch neural network (FF and LSTM), policy, critic implementation.
* logs: contains the logs of the training i.e. each experimtent.
* exp_confs: contains the experiment configuration files for training and testing.
* trng_tools: contains the tools for training.
* documentation: contains the documentation for the codebase.
* anlys_logs: contains the logs of any analysis genreally. (like mmode_transition_test2, qp_plans, etc).
* main.py: the primary file to deploy trainings and run validation of the experiments thrrough predefined tests for quantifying performance.
* util.py: contains the utility functions required by main.py.
* record_policy.py: the file to run any custom tests with a single policy (like user defined mode sequences, each mode of a policy, etc).
* template.job: the template job file for deploying trainings on the cluster.
* README.md: this file.
* ttbd_command.txt: a example command to run a single training locally, to test any new to environment that may affect the training.