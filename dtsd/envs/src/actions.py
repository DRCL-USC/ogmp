
from dtsd.envs.src.transformations import euler_to_quat,quat_to_euler,quat_to_mat
from dtsd.envs.src.misc_funcs import *
import numpy as np

# for hardware roperties
LEG_NAMES = ['L','R']
JNT_NAMES = ['hip','hip2','thigh','calf','toe']

def predict_vector(env,action_subset):
    return np.zeros_like(env.sim.data.ctrl)

def pd_targets(env,action_subset):
    
    target = action_subset + np.array(env.model_prop[env.exp_conf['robot']]['jpos_nominal'])    
    if exists_and_true('fixed_nominal_jpos',env.exp_conf):
        target = np.array(env.model_prop[env.exp_conf['robot']]['jpos_nominal'])    
    torques = np.zeros_like(env.sim.data.ctrl)    
    for i,(jci,jpi,jvi) in enumerate(
                                            zip(
                                                env.model_prop[env.exp_conf['robot']]['ids']['jctrl'],                                                
                                                env.model_prop[env.exp_conf['robot']]['ids']['jpos'],
                                                env.model_prop[env.exp_conf['robot']]['ids']['jvel'],
                                                )
                                        ):
        torques[jci] =  env.exp_conf['p_gain'][i]*(target[i] - env.sim.data.qpos[jpi]) \
                      + env.exp_conf['d_gain'][i]*(        0 - env.sim.data.qvel[jvi])

    return torques

def direct_pd_targets(env,action_subset):
    return action_subset