import os,sys
sys.path.append('./')
from oracles.models import biped_model
import numpy as np
from dtsd.envs.src import transformations

class oracle:
    def __init__(self,terrain_map_resolution = [0.001,0.001]):
        self.ddt = 0.0125
        self.tf = 1.0
        self.dt = 0.03
        self.g = -9.8
        self.model = biped_model()
    
    def _index2coord_(self,index = [0,0]):
       raise NotImplementedError
    
    def _terrain_map2list_(self,terrain_map):
        raise NotImplementedError

    def _generate_reference_(self,mode_name,internal_params):
        raise NotImplementedError
    
    def get_traj(self,mode_name,input):  #observations are proprioceptive or exprioceptive info, dictionary of data 
        raise NotImplementedError

    def _qpos_vel_traj_(self,x_sol):
        x_sam = x_sol[:,:int(self.tf/self.ddt):int(self.dt/self.ddt)]
        N = x_sam.shape[1]
        base_pos_traj = x_sam[3:6,:].T
        base_ori_traj = np.zeros((N,4))

        for i in range(N):
            base_ori_traj[i,:] = transformations.euler_to_quat(x_sam[:3,i])
        base_tvel_traj = x_sam[9:,:].T
        base_avel_traj = x_sam[6:9,:].T

        jpos_traj = np.zeros((N,10))
        jvel_traj = np.zeros((N,10))

        qpos_traj = np.concatenate((base_pos_traj, base_ori_traj,jpos_traj), axis=1)
        qvel_traj = np.concatenate((base_tvel_traj, base_avel_traj,jvel_traj), axis=1)

        return qpos_traj,qvel_traj

    def qpos_vel2state(self,qpos,qvel):
        state = np.zeros((13,1))
        
        state[:3,0] = transformations.quat_to_euler(qpos[3:7],ordering = 'ZYX')
        state[3:6,0] = qpos[0:3]
        state[6:9,0] = qvel[3:6]
        state[9:12,0] = qvel[0:3]
        state[12] = self.g
        return state



        