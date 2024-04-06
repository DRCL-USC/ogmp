from dtsd.envs.sim.ogpo_biped_base import *

# ORACLE CONSTANTS FOR THE PARKOUR ENVIRONMENT
ORACLE_X0_NOMINAL =   [
                          0.0, 0.0, 0.0,
                          0.0, 0.0, 0.5,
                          0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0,
                        ]

ORACLE_X0_LABELS = [
                      'roll', 'pitch', 'yaw', 
                      'x', 'y', 'z',
                      'roll_dot', 'pitch_dot', 'yaw_dot',
                      'x_dot', 'y_dot', 'z_dot',

                      ] 

class env(ogpo_biped_base):
    def __init__(self, exp_conf_path='./exp_confs/default.yaml'):
        super().__init__(exp_conf_path)

    def generate_reference(self):

        # scan the terrain infront of the robot
        terrain_map,_ = self.scan_terrain_xlen_infront(xlen=self.exp_conf['oracle']['scan_xlen_infront'],
                                                        return_start_pos=True)
        terrain_map = terrain_map.T

        # get current state
        x0 = self.oracle.qpos_vel2state(
                                        np.copy(self.sim.data.qpos),
                                        np.copy(self.sim.data.qvel)
                                        )
                
        # only send selcted feedback
        for x0_n, sf in zip(ORACLE_X0_NOMINAL,ORACLE_X0_LABELS):
            if sf not in self.exp_conf['oracle']['state_feedback']:
                x0[ORACLE_X0_LABELS.index(sf)] = x0_n

        feedback = {
                    'terrain_scan_x':terrain_map,
                    'delta_tpos':np.array([[1.0],[0]]),
                    'x0':x0,
                    }
        x_sol, u_sol,qpos,qvel, = self.oracle.get_traj('fbg', feedback)

        # for debugging
        # self.oracle.plotter(x_sol, u_sol)
        # self.oracle.animate(x_sol,u_sol,fname=None)  
        
        self.curr_ref_traj = biped_trajectory_preview(qpos, qvel, self.dt)

    def check_terminations(self, current_reward, verbose=False):
        
        done, _ = super().check_terminations(current_reward, verbose) # returns if done already
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

            # 'base_pos_x_ref_error_thresh'
            target = target_base_pose[0:3] # reference motion pos
            error = np.abs(base_pos[0]-target[0])
            if error>self.exp_conf['terminations']['base_pos_x_ref_error_thresh']:
                if verbose: print('done','base_pos_x_ref_error_thresh')
                return True, 'base_pos_x_ref_error_thresh'
            
            # base_min_height_thresh (only used in n_rollout test)
            if 'base_height_min_thresh' in self.exp_conf['terminations'].keys():
                terrain_height = self.sim.get_terrain_height_at(base_pos)
                height = base_pos[-1] - terrain_height
                if height < self.exp_conf['terminations']['base_height_min_thresh']: 
                    if verbose: print('done','base_height_min_thresh')
                    return True, 'base_height_min_thresh'


        return False, None

    def generate_task_variant(self):
      
        if exists_not_none('track_x_start',self.exp_conf['task']):
            track_xlen = self.exp_conf['task']['track_x_start']
        else:
            track_xlen = 0.0

        # choose the starting mode
        curr_mode_id =  np.random.randint(
                                        low=0,
                                        high=len(self.exp_conf['task']['modes'].keys())
                                        )
        while track_xlen < self.exp_conf['task']['track_x_length']:
            # get the distribution of the current mode
            te_prob_dist = self.transition_samplers[curr_mode_id].return_prob()
            # sample the nex mode
            curr_mode_name = np.random.choice(
                                                list(self.exp_conf['task']['modes'].keys()),
                                                p=te_prob_dist        
                                        )
            # update mode id
            curr_mode_id = list(self.exp_conf['task']['modes'].keys()).index(curr_mode_name)

            # chose the mode variant
            curr_mode_dict = self.exp_conf['task']['modes'][curr_mode_name]
            # sampl the mode paramters
            if 'discrete' in curr_mode_dict['param_dist']['type']: 
                curr_mode_param = np.random.choice(curr_mode_dict['param_dist']['points'])
            elif curr_mode_dict['param_dist']['type'] == 'continuous':
                curr_mode_param = np.random.uniform(
                                                    low=curr_mode_dict['param_dist']['support'][0], 
                                                    high=curr_mode_dict['param_dist']['support'][-1] 
                                                    )
            
            if exists_and_true('manipulate_terrain',curr_mode_dict):
                # for gap and blocks
                gb_start = track_xlen+curr_mode_param[0]
                gb_end = track_xlen+curr_mode_param[0]+curr_mode_param[1]
                gb_height = curr_mode_param[2]
                
                self.sim.generate_terrain_plateau(
                                                gb_start+0.05,
                                                gb_end+0.05,
                                                -0.5,0.5,
                                                gb_height
                                                )                
                track_xlen += curr_mode_param[0]+curr_mode_param[1]
            else:
                # for flat
                track_xlen += curr_mode_param[0] # goal_x

    def scan_terrain_xlen_infront(self,xlen=1.0,return_start_pos=False):

        base_pos = self.get_robot_base_pos()
        terrain_map = self.sim.get_terrain_infront(
                                                    pos=base_pos,
                                                    halfwidth_x=xlen,
                                                    halfwidth_y=0.08
                                                    )
        x_half_index = int(0.5*terrain_map.shape[0])
        terrain_map = terrain_map[x_half_index:,0:1]

        if return_start_pos:
            return terrain_map,base_pos

        return terrain_map

    def set_markers_for_terrain_infront(
                                        self,
                                        base_pos,
                                        halfwidth_y,
                                        halfwidth_x,
                                        density_x=1.0,
                                        density_y=1.0,
                                        ):

        i_start = int( (base_pos[0]) /self.sim.hfield_x_index2len + 0.5*self.sim.model.hfield_nrow[0])
        i_end   = int( (base_pos[0]+halfwidth_x) /self.sim.hfield_x_index2len + 0.5*self.sim.model.hfield_nrow[0])
        
        j_start = int( (base_pos[1]-halfwidth_y) /self.sim.hfield_y_index2len + 0.5*self.sim.model.hfield_ncol[0])
        j_end = int( (base_pos[1]+halfwidth_y) /self.sim.hfield_y_index2len + 0.5*self.sim.model.hfield_ncol[0])

        
        for i in range(i_start,i_end, int(1/density_x)):
            for j in range(j_start,j_end, int(1/density_y)):
                
                x_at_i = (i - 0.5*self.sim.model.hfield_nrow[0])*self.sim.hfield_x_index2len 
                y_at_j = (j - 0.5*self.sim.model.hfield_ncol[0])*self.sim.hfield_y_index2len 
                z_at_ij = np.copy(self.sim.hfield_scale*self.sim.model.hfield_data[j*self.sim.model.hfield_ncol[0] + i])
                
                self.sim.vis_markers.append(
                                                {
                                                'pos':np.array([x_at_i,y_at_j,z_at_ij]),
                                                'size':np.array([0.02,0.02,0.001]),
                                                'rgba':np.array([0.0,1.0,0.0,1.]),
                                                }
                                            )
