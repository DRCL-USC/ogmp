# OGMP: Oracle Guided Multimodal Policies for Agile and Versatile Robot Control

Codebase accompanying the paper [OGMP: Oracle Guided Multimodal Policies for Agile and Versatile Robot Control](https://arxiv.org/abs/2403.04205). For support please raise an issue here or contact the authors.

<p align="center">
   <img width="400" height="226" src="media/parkour.gif">
   <img width="400" height="226" src="media/dive_all4.gif">
</p>

## Installation

Make a virtual environment,

    python3 -m venv ogmp_env
    source ogmp_env/bin/activate

Clone the repo,

    git clone --depth 1 https://github.com/DRCL-USC/ogmp

To install the dependencies, run:
    
    pip3 install -r requirements.txt

Install old version of torch (new may work but is not tested)
    
    pip3 install torch==1.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

tested in python 3.8.10

## Usage

To test the policy in paper, for each task, run
    
    # for best parkour policy
    python3 test.py --tstng_conf_path ./exp_confs/parkour_test.yaml --render_onscreen

    # for best dive policy
    python3 test.py --tstng_conf_path ./exp_confs/dive_test.yaml --render_onscreen

Similarly to train the best policy from the paper, run 

    # for parkour
    python3 train.py --exp_conf_path ./exp_confs/parkour.yaml --recurrent --logdir ./logs/

    # for dive
    python3 train.py --exp_conf_path ./exp_confs/dive.yaml --recurrent --logdir ./logs/

To run the analyses from the paper, run

    # for the sample mean of agility metrics and in-domain paramters
    python3 analysis/n_rollout_test.py 

    # for LMSR on flat ground
    python3 analysis/flat_ground_lmsr_test.py 

    # for LMSR at transition
    python3 analysis/transition_lmsr_test.py 

The corresponding config file for each analyisis is in `./exp_confs/` folder. All the analyses files deploy multiprocessing, where each process will have a copy of a lstm policy and hence could be intensive. Set `nop` as per your system's capability to minimize time (default is 3 tested in a 32 Core, 64 GB RAM machine). Resulting plots and ogs will be saved in `./results/<experiment_name>/<variant_id>`.



## High-level overview

* algos: contains the custom ppo implementation from [link](https://github.com/osudrl/RSS-2020-learning-memory-based-control)
* analysis: contains the code for analysis presenteed in the paper
* nn: contains the custom torch neural network (FF and LSTM), policy, critic implementation from [link](https://github.com/osudrl/RSS-2020-learning-memory-based-control)
* logs: contains the training logs, policies and encoders.
* dtsd: contains the environments
* exp_confs: contains the experiment configuration files for training and testing.
* train.py: file to train policies.
* test.py: file to test policies.

## Recreating results from the paper

<img src="media/results_recreated.jpg" align="right" width="300"/>
Since the paper, the codebase has been cleaned and made modular for easy usage. This results in minor changes in the training convergence (as in the figure) and analyses results, but qualitatively the policy's performance is indistinguishable and assertions of the analyses hold.

<br clear="left"/>

## Citation

If you find this code useful, consider citing:

```
    @misc{krishna2024ogmp,
      title={OGMP: Oracle Guided Multimodal Policies for Agile and Versatile Robot Control}, 
      author={Lokesh Krishna and Nikhil Sobanbabu and Quan Nguyen},
      year={2024},
      eprint={2403.04205},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
    }
```
