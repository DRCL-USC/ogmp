
import numpy as np
from dtsd.envs.src.misc_funcs import *

def prediction_error(env):   
    # NOTE: GENERALIZED YET
    base_tvel = env.sim.data.qvel[
                                    env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'][0]:
                                    1+env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'][-1]
                                ]
    base_avel = env.sim.data.qvel[
                                    env.model_prop[env.exp_conf['robot']]['ids']['base_avel'][0]:
                                    1+env.model_prop[env.exp_conf['robot']]['ids']['base_avel'][-1]
                                ]
    env.get_toe_contact_state()
    
    pv_ft_idx = env.exp_conf['rewards']['prediction_vector_fromto_idx']
    
    x_true = np.concatenate([base_tvel,base_avel,env.curr_toe_contact_state])
    x_pred = env.curr_action[pv_ft_idx[0]:pv_ft_idx[-1]]
    # print('#################')
    # print('true:',x_true.round(2))
    # print('pred:',x_pred.round(2))
    # print("error:",np.linalg.norm(x_true-x_pred))   

    prediction_error = env.exp_conf['rewards']['prediction_error_exp_weight'] * np.linalg.norm(x_true-x_pred)
    return np.exp(-prediction_error)

def base_tvel_x_const(env):
    actual = env.sim.data.qvel[
                                env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'][0]:
                                1+env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'][-1]          
                                ]
    target = env.exp_conf['rewards']['base_tvel_x_const_target']
    base_tvel_x_const_error = env.exp_conf['rewards']['base_tvel_x_const_exp_weight'] * np.abs(actual[0]-target)
    return np.exp(-base_tvel_x_const_error)

def base_ori_ref_error(env):
    actual_q = env.sim.data.qpos[
                                env.model_prop[env.exp_conf['robot']]['ids']['base_ori'][0]:
                                1+env.model_prop[env.exp_conf['robot']]['ids']['base_ori'][-1]          
                                ]
    
    target_q = env.get_ref_state(env.phase)[0][3:7] # reference motion ori


    # Let's say the two versors describing the two orientations are q1 and q2,
    # q1q2=(w1,x1,y1,z1)=(w2,x2,y2,z2)
    # and difference in the two orientations is qΔ,
    # q2=qΔq1⟺qΔ=q2q−11
    # Note that we use quaternion operations here: Hamilton product for multiplying the two orientations. 
    # For unit quaternions, the reciprocal is the same as conjugate; and is just negating the vector components (x, y, and z).

    # (Note that negating all four components of the orientation does not change the orientation, it just switches between the "long way" and "short way" 
    # around the great circle on the unit sphere. If you interpolate between two quaternions, if their dot product q1q2+x1x2+y1y2+z1z2 is negative, 
    # you interpolate the long way around; negating all four components of one of the quaternions lets you interpolate the short way around instead,
    #  without affecting the start and end orientations at all.)

    # This means that the error orientation, or the rotation needed to transform orientation q1 to q2 is qΔ,
    # qΔ=(qΔ,xΔ,yΔ,zΔ)

    base_ori_ref_error = env.exp_conf['rewards']['base_ori_error_exp_weight'] * (1 - np.inner(actual_q, target_q) ** 2)      
    return np.exp(-base_ori_ref_error)

def base_pos_ref_error(env):
    actual = env.sim.data.qpos[
                                env.model_prop[env.exp_conf['robot']]['ids']['base_pos'][0]:
                                1+env.model_prop[env.exp_conf['robot']]['ids']['base_pos'][-1]          
                                ]

    target = env.get_ref_state(env.phase)[0][0:3] # reference motion pos
    
    base_pos_ref_error = env.exp_conf['rewards']['base_pos_error_exp_weight'] * np.linalg.norm(target-actual)
    
    return np.exp(-base_pos_ref_error)

def base_avel_ref_error(env):
    actual = env.sim.data.qvel[
                                env.model_prop[env.exp_conf['robot']]['ids']['base_avel'][0]:
                                1+env.model_prop[env.exp_conf['robot']]['ids']['base_avel'][-1]          
                                ]
    target = env.get_ref_state(env.phase)[1][3:6] # reference motion avel
    base_avel_ref_error = env.exp_conf['rewards']['base_avel_error_exp_weight'] * np.linalg.norm(target-actual)
    
    return np.exp(-base_avel_ref_error)

def base_tvel_ref_error(env):
    actual = env.sim.data.qvel[
                                env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'][0]:
                                1+env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'][-1]          
                                ]
    target = env.get_ref_state(env.phase)[1][0:3] # reference motion tvel      
    base_tvel_ref_error = env.exp_conf['rewards']['base_tvel_error_exp_weight'] * np.linalg.norm(target-actual)
    
    return np.exp(-base_tvel_ref_error)

def toe_contact_error(env):

    ltoe_contact, _ = env.sim.contact_bw_bodies(body1='terrain',body2='L_toe')
    rtoe_contact, _ = env.sim.contact_bw_bodies(body1='terrain',body2='R_toe')              
    
    ltoe_contact = 1 if ltoe_contact else 0
    rtoe_contact = 1 if rtoe_contact else 0

    actual = [ltoe_contact,rtoe_contact]
    _,_,target = env.get_ref_state(env.phase,send_contact=True) # reference motion ori

    toe_contact_ref_error = env.exp_conf['rewards']['toe_contact_error_exp_weight'] * np.linalg.norm(target-actual)
    # print(target, actual,toe_contact_ref_error,np.exp(-toe_contact_ref_error))
    return np.exp(-toe_contact_ref_error)

def joint_error_nominal(env):
    joint_error = 0
    ref_pos = env.sim.model.qpos0[ 
                                env.model_prop[env.exp_conf['robot']]['ids']['qpos_slice'][0]:
                                env.model_prop[env.exp_conf['robot']]['ids']['qpos_slice'][1]
                                ]

    for i, j in enumerate(env.model_prop[env.exp_conf['robot']]['ids']['jpos']):
        target = ref_pos[j]
        actual = env.sim.data.qpos[j]
        joint_error += 30 * env.exp_conf['rewards']['joint_error_exp_weight'][i] * (target - actual) ** 2

    if 'only_upon_toes_contact' in env.exp_conf['rewards'].keys() and \
        'joint_error_nominal' in env.exp_conf['rewards']['only_upon_toes_contact']:
        
        ltoe_contact, _ = env.sim.contact_bw_bodies(body1='terrain',body2='L_toe')
        rtoe_contact, _ = env.sim.contact_bw_bodies(body1='terrain',body2='R_toe')

        if not(ltoe_contact or rtoe_contact):
            # in fight
            joint_error = np.inf # reward should be zero else
        
    return np.exp(-joint_error)

def action_smoothness(env):
    action_smoothness = env.exp_conf['rewards']['action_smoothness_exp_weight'] * np.linalg.norm(env.curr_action-env.prev_action) 
    return np.exp(-action_smoothness)

def ctrl_smoothness(env):
    ctrl_smoothness = env.exp_conf['rewards']['ctrl_smoothness_exp_weight'] * np.linalg.norm(env.sim.data.ctrl[:]-env.prev_ctrl)
    return np.exp(-ctrl_smoothness)

def ctrl_mag(env):
    ctrl_mag = env.exp_conf['rewards']['ctrl_mag_exp_weight'] * np.linalg.norm(env.sim.data.ctrl[:])
    return np.exp(-ctrl_mag)

def ctrl_mag_weighted(env):
    weighted_l2_norm = np.sqrt(
                                weighted_quadratic(
                                                    x=env.sim.data.ctrl[:],
                                                    w=np.diag(env.exp_conf['rewards']['ctrl_mag_weighted_r'])
                                                    )
                                )
    ctrl_mag = env.exp_conf['rewards']['ctrl_mag_weighted_exp_weight'] * np.linalg.norm(weighted_l2_norm)
    return np.exp(-ctrl_mag)

def action_smoothness_nrm(env):
    return env.exp_conf['rewards']['action_smoothness_nrm_weight'] * np.linalg.norm(env.curr_action-env.prev_action)

def ctrl_smoothness_nrm(env):
    return env.exp_conf['rewards']['ctrl_smoothness_nrm_weight'] * np.linalg.norm(env.sim.data.ctrl[:]-env.prev_ctrl)

def ctrl_mag_nrm(env):
    return env.exp_conf['rewards']['ctrl_mag_nrm_weight'] * np.linalg.norm(env.sim.data.ctrl[:])

def penalize_non_toes_terrain_contact(env):

    for i in range(env.sim.model.nbody):
        body_name = env.sim.obj_id2name(obj_id=i,type='body')
        if body_name not in ['L_toe','R_toe','l_toe', 'r_toe']:
            contact, _ = env.sim.contact_bw_bodies(body1='terrain',body2=body_name)
        if contact:
            return env.exp_conf['rewards']['non_toes_terrain_contact_penalty']    
    return 0

def penalize_not_allowed_bodies_terrain_contact(env):
    for i in range(env.sim.model.nbody):
        body_name = env.sim.obj_id2name(obj_id=i,type='body')
        if body_name not in env.exp_conf['rewards']['allowed_bodies_terrain_contact']:
            contact, _ = env.sim.contact_bw_bodies(body1='terrain',body2=body_name)
        if contact:
            return env.exp_conf['rewards']['not_allowed_bodies_terrain_contact_penalty']    
    return 0

def penalize_poor_contact(env):        

    poor_contact_penatly = 0
    for body_name in env.exp_conf['rewards']['poor_contact_bodies']:
        tcn, tcf = env.sim.all_contact_bw_bodies(body1='terrain',body2=body_name)
        if tcn > 0 and  tcn < env.exp_conf['rewards']['poor_contact_thresh']:
            poor_contact_penatly += env.exp_conf['rewards']['poor_contact_penalty']

    return poor_contact_penatly

def incentivize_gait_v1(env):

    # from sensor data
    ltoe_cfrc = env.sim.total_contact_force_bw_bodies(body1='terrain',body2='l_toe')
    rtoe_cfrc = env.sim.total_contact_force_bw_bodies(body1='terrain',body2='r_toe')

    ltoe_vel = env.sim.get_sensordata('l_toe_vel')
    rtoe_vel = env.sim.get_sensordata('r_toe_vel')

    if env.this_epi_nstep < env.initial_balance_steps:
        left_contact_schedule =  1
        right_contact_schedule = 1
    else:
        time_now = (env.this_epi_nstep - env.initial_balance_steps) * env.dt
        # a two phase gait
        left_norm_t = time_now % env.exp_conf['rewards']['gait_period']
        right_norm_t = (time_now + env.exp_conf['rewards']['phase_off_right_leg']) % env.exp_conf['rewards']['gait_period']

        left_contact_schedule =  1 if left_norm_t < (env.exp_conf['rewards']['gait_period'] * env.exp_conf['rewards']['stance_percentage']) else 0
        right_contact_schedule = 1 if right_norm_t < (env.exp_conf['rewards']['gait_period'] * env.exp_conf['rewards']['stance_percentage']) else 0

    # if in contact penalize velocity, else penalize force
    ltoe_reward = -env.exp_conf['rewards']['contact_vel_exp_weight']*np.linalg.norm(ltoe_vel) \
        if left_contact_schedule else -env.exp_conf['rewards']['contact_frc_exp_weight']*np.linalg.norm(ltoe_cfrc)
    
    rtoe_reward = -env.exp_conf['rewards']['contact_vel_exp_weight']*np.linalg.norm(rtoe_vel) \
        if right_contact_schedule else -env.exp_conf['rewards']['contact_frc_exp_weight']*np.linalg.norm(rtoe_cfrc)
    
    # print('########### started:', not env.this_epi_nstep < env.initial_balance_steps, env.initial_balance_steps)
    # print(left_contact_schedule,'ltoe_reward:',np.exp(ltoe_reward).round(2), np.linalg.norm(ltoe_cfrc).round(2),np.linalg.norm(ltoe_vel).round(2))
    # print(right_contact_schedule,'rtoe_reward:',np.exp(rtoe_reward).round(2), np.linalg.norm(rtoe_cfrc).round(2),np.linalg.norm(rtoe_vel).round(2))
    
    return np.exp(ltoe_reward) + np.exp(rtoe_reward)