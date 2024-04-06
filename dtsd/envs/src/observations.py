
from dtsd.envs.src.transformations import euler_to_quat,quat_to_euler,quat_to_mat
from dtsd.envs.src.misc_funcs import *
import torch
import numpy as np

# observation functions:
def clock_osu(env):

    return( [np.sin(2 * np.pi *  env.phase / env.phaselen),
                np.cos(2 * np.pi *  env.phase / env.phaselen)] )

def clock(env):

    return( [np.sin(2 * np.pi *  env.phase / env.phaselen),
                np.cos(2 * np.pi *  env.phase / env.phaselen)] )

def robot_state(env):

    base_pos = env.get_robot_base_pos()
    terrain_height = env.sim.get_terrain_height_at(base_pos)
    base_height_frm_support_plane = env.sim.data.qpos[env.model_prop[env.exp_conf['robot']]['ids']['base_pos'][-1]] - terrain_height
    
    base_ori =  np.take(env.sim.data.qpos,env.model_prop[env.exp_conf['robot']]['ids']['base_ori'])
    base_tvel = np.take(env.sim.data.qvel,env.model_prop[env.exp_conf['robot']]['ids']['base_tvel'])
    base_avel = np.take(env.sim.data.qvel,env.model_prop[env.exp_conf['robot']]['ids']['base_avel'])        
    jpos = np.take(env.sim.data.qpos,env.model_prop[env.exp_conf['robot']]['ids']['jpos'])
    jvel = np.take(env.sim.data.qvel,env.model_prop[env.exp_conf['robot']]['ids']['jvel'])

    # transform to body frame
    rotmat_w2b = quat_to_mat(base_ori)[0:3,0:3]     
    base_tvel = np.dot(rotmat_w2b,base_tvel)

    obs_data = [
                [base_height_frm_support_plane],
                base_ori,
                jpos,
                base_tvel,
                base_avel,
                jvel,
                ]

    robot_state = np.concatenate(obs_data).ravel()
    return robot_state

def mode_latent(env):

    ref_traj_offset_qpos = np.copy(env.curr_ref_traj.qpos[:env.exp_conf['oracle']['prediction_horizon'],:])
    ref_traj_offset_qvel = np.copy(env.curr_ref_traj.qvel[:env.exp_conf['oracle']['prediction_horizon'],:])

    # offsets
    if exists_not_none('offset_x0',env.exp_conf['mode_encoder']):

        for i,sf in enumerate(['x','y','z']):
            if sf in env.exp_conf['mode_encoder']['offset_x0']:
                ref_traj_offset_qpos[:,i] -= env.curr_ref_traj.qpos[0,i]

        change_rpy = False
        for i,sf in enumerate(['roll','pitch','yaw']):
            if sf in env.exp_conf['mode_encoder']['offset_x0']:
                change_rpy = True
                print('changing rpy')

        if change_rpy:
            curr_ref_traj_rpy0 = quat_to_euler(env.curr_ref_traj.qpos[0,3:7])
            
            for i in range(len(env.curr_ref_traj.qpos)):
                ref_traj_rpy = quat_to_euler(ref_traj_offset_qpos[i,3:7])
                ref_traj_rpy -= curr_ref_traj_rpy0
                ref_traj_offset_qpos[i,3:7] = euler_to_quat(ref_traj_rpy)

        for i,sf in enumerate([
                                'x_dot','y_dot','z_dot',
                                'roll_dot','pitch_dot','yaw_dot'
                            ]):
            if sf in env.exp_conf['mode_encoder']['offset_x0']:
                ref_traj_offset_qvel[:,i] -= env.curr_ref_traj.qvel[0,i]


    # select the necessary inputs          
    if exists_and_is_equal('input_type',env.exp_conf['mode_encoder'],'base_only'):
        enc_input =  np.append(
                                ref_traj_offset_qpos[:,0:7],
                                ref_traj_offset_qvel[:,0:6],
                                axis=1
                            ).astype('float32')

    else:
        # default full state
        enc_input =  np.append(
                                ref_traj_offset_qpos,
                                ref_traj_offset_qvel,
                                axis=1
                            ).astype('float32')

    ref_state_traj = torch.from_numpy(enc_input)

    # mode_latent = env.mode_enc(ref_state_traj)
    # env.curr_mode_latent  = mode_latent.detach().numpy()
    env.curr_mode_latent  = env.get_mode_latent_from_traj(ref_state_traj)
    # print(env.curr_mode_latent)
    return  env.curr_mode_latent

def custom_latent(env):
    if  env.curr_mode_latent is None:
        env.curr_mode_latent = np.zeros(2)
    return  env.curr_mode_latent

def terrain_xlen_infront(env):
    return env.scan_terrain_xlen_infront(xlen=env.exp_conf['observations']['terrain_xlen_infront']).squeeze()

def terrain_around(env):
    return env.scan_terrain_around(xlen=env.exp_conf['observations']['terrain_xlen_around'],
                                    ylen=env.exp_conf['observations']['terrain_ylen_around']).squeeze()