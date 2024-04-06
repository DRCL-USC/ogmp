from dtsd.envs.sim.ogpo_biped_base import *

class env(ogpo_biped_base):
    def __init__(self, exp_conf_path='./exp_confs/default.yaml'):
        super().__init__(exp_conf_path)

    def step(self, action):
        sup_return = super().step(action)
        self.exp_conf['oracle']['prediction_horizon'] = self.phaselen
        return sup_return
    
    def reset(self):
        sup_return =  super().reset()
        self.exp_conf['oracle']['prediction_horizon'] = self.phaselen
        return sup_return

    def check_terminations(self, current_reward, verbose=False):
        
        done,_ = super().check_terminations(current_reward, verbose) # returns if done already

        if done:
            return done, _
        else:
            # get data to check terminations
            base_pos = self.get_robot_base_pos()
            target_base_pose = self.get_ref_state(self.phase)[0]

            # 'total_reward_min_thresh'
            if current_reward < self.exp_conf['terminations']['total_reward_min_thresh']:
                if verbose: print('done','total_reward_min_thresh')
                return True, 'total_reward_min_thresh'

            # 'base_pos_z_ref_error_thresh'
            target = target_base_pose[0:3] # reference motion pos
            error = np.abs(base_pos[2]-target[2])
            if error>self.exp_conf['terminations']['base_pos_z_ref_error_thresh']:
                if verbose: print('done','base_pos_z_ref_error_thresh')
                return True, 'base_pos_z_ref_error_thresh'

        return False, None

    def generate_reference(self):
        
        if self.this_epi_nstep > 0:
            # settle
            pos_init = self.curr_ref_traj.qpos[-1,0:3]
            rpy_init = quat_to_euler(self.curr_ref_traj.qpos[-1,3:7])
            x0 = np.array([[
                            0.0, 0.0, rpy_init[2], 
                            pos_init[0], pos_init[1], 0.5, 
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 
                            -9.8]]).T
            delta_apos = np.array([[0.0],[0.0],[0.0]])
            self.curr_mode['name'] = 'land'
        else:
            # flip
            x0 = np.array([[
                                  0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.5+ self.curr_mode['param'][0], 
                                  0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 
                                -9.8
                            ]]).T
        
            # choose the mode variant            
            if self.curr_mode['name'] == 'pitch_flip':
                if self.curr_mode['param'][1] > 0:
                    # front flip 
                    delta_apos = np.array([[0.0],[2*np.pi],[0.0]])
                else:
                    # back flip
                    delta_apos = np.array([[0.0],[-2*np.pi],[0.0]])
            elif self.curr_mode['name'] == 'roll_flip':
                if self.curr_mode['param'][1] > 0:
                    # left flip
                    delta_apos = np.array([[2*np.pi],[0.0],[0.0]])
                else:
                    # right flip
                    delta_apos = np.array([[-2*np.pi],[0.0],[0.0]])

        feedback = {
                    'x0':x0,
                    'hi':self.curr_mode['param'][0],
                    'hf':0,
                    "delta_apos":delta_apos,
                    }
            
        x_sol, u_sol,qpos, qvel, = self.oracle.get_traj(self.curr_mode['name'],feedback)
        # self.oracle.plotter(x_sol, u_sol)
        self.curr_ref_traj = biped_trajectory_preview(qpos, qvel, self.dt)
  
    def generate_task_variant(self):

        # choose a mode
        curr_mode_id =  np.random.randint(
                                            low=0,
                                            high=len(self.exp_conf['task']['modes'].keys())
                                        )
        curr_mode_name = list(self.exp_conf['task']['modes'].keys())[curr_mode_id]
        curr_mode_dict = self.exp_conf['task']['modes'][curr_mode_name]
        if 'discrete' in curr_mode_dict['param_dist']['type']: #== 'discrete':
            curr_mode_param = np.random.choice(curr_mode_dict['param_dist']['points'])
        elif curr_mode_dict['param_dist']['type'] == 'grid':
            curr_mode_param = np.random.uniform(
                                                    low=curr_mode_dict['param_dist']['support'][0], 
                                                    high=curr_mode_dict['param_dist']['support'][-1] 
                                                )
        
        h_blk = curr_mode_param[0]
        self.sim.generate_terrain_plateau(
                                            -0.2,
                                            0.2,
                                            -0.2,
                                            0.2,
                                            h_blk
                                            )
        
        self.curr_mode = {
                            'name':curr_mode_name,
                            'param':curr_mode_param
                        }
        

