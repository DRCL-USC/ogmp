import numpy as np
from scipy.interpolate import interp1d

# NOTE: need to add downsampling
class biped_trajectory:
    def __init__(
                    self, 
                    traj_filepath,
                    traj_slice_t = None,
                    reference_dt = None,
                    add_jpos_offsets = True,

                ):

        traj_data = np.load(traj_filepath,allow_pickle=True)

        #NOTE: later should obtain from robot properties
        joint_offsets = np.array([
                                0,0,np.pi/4,-np.pi/2,np.pi/4,
                                0,0,np.pi/4,-np.pi/2,np.pi/4,
                            ])
        if not add_jpos_offsets:
            joint_offsets = np.zeros_like(joint_offsets)

        com_state_len = 7
        
        
        if 'cont_seq' in traj_data.files:
            cont_seq = traj_data['cont_seq']
        qpos_traj = traj_data['qpos_traj']
        # add joint offsets bw simulation and trajector optimisation
        try:
            qpos_traj[:,com_state_len:] -= joint_offsets
        except:
            pass
        qvel_traj = traj_data['qvel_traj']


        traj_time = traj_data['time']


        # slicing in original time 
        if traj_slice_t != None:
            lower_bound_i = np.where(traj_time==traj_slice_t[0])[0][0]
            upper_bound_i = np.where(traj_time==traj_slice_t[1])[0][0]
            qpos_traj = qpos_traj[lower_bound_i:upper_bound_i,:]
            qvel_traj = qvel_traj[lower_bound_i:upper_bound_i,:]
            traj_time = traj_time[lower_bound_i:upper_bound_i]
            if 'cont_seq' in traj_data.files:
                cont_seq = cont_seq[lower_bound_i:upper_bound_i]

        # interpolating new frequency
        if reference_dt != None:
            traj_dt = traj_time[1] - traj_time[0]

            # print('Orignal Trajectory\'s dt:',traj_dt)
            # print('Scaled Trajectory to dt:',reference_dt)

            n_samples = traj_time.shape[0]
            n_samples_new = int(traj_dt/reference_dt)*n_samples

            qpos_traj_new = []
            qvel_traj_new = []
            cont_seq_new = []

            traj_time_new = np.linspace(traj_time[0], traj_time[-1], num=n_samples_new, endpoint=True)
            for i,row in enumerate(qpos_traj.T):
                
                f = interp1d(traj_time, row)
                q_new = f(traj_time_new)
                qpos_traj_new.append(q_new)
                qpos_traj_new.append(q_new)
                
                if i < qvel_traj[0].shape[0]:
                    df = interp1d(traj_time, qvel_traj.T[i] )
                    dq_new = df(traj_time_new)
                    qvel_traj_new.append(dq_new)
                if 'cont_seq' in traj_data.files:
                    if i < cont_seq[0].shape[0]:
                        df = interp1d(traj_time, cont_seq.T[i] )
                        dc_new = df(traj_time_new)
                        cont_seq_new.append(dc_new)




            qpos_traj = np.array(qpos_traj_new,dtype=list).T
            qvel_traj = np.array(qvel_traj_new,dtype=list).T
            if 'cont_seq' in traj_data.files:
                cont_seq = np.array(cont_seq_new,dtype=list).T

            traj_time = traj_time_new


        # set data
        self.time = traj_time
        self.qpos = qpos_traj
        self.qvel = qvel_traj
        if 'cont_seq' in traj_data.files:
            self.cont_seq = cont_seq

    def __len__(self):
        return len(self.time)
    
class biped_trajectory_preview:
    def __init__(
                    self, 
                    qpos_traj,
                    qvel_traj,
                    t_traj=None,
                    dt=0.03,
                ):

        

        # set data
        self.time = dt*np.arange(qpos_traj.shape[0])
        self.qpos = qpos_traj
        self.qvel = qvel_traj
    def __len__(self):
        return len(self.time)    

    

if __name__ == '__main__':
    traj_filepath = "./dtsd/rsc/trajectories/drcl_kdto/npzs/biped_walk.npz"
    reference_dt = 0.002
    traj_object = biped_trajectory(
                                    traj_filepath,
                                    reference_dt
                                    )


    print(traj_object.time.shape)
    print(traj_object.qpos.shape)
    print(traj_object.qvel.shape)
