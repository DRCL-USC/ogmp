from dtsd.envs.src.transformations import euler_to_quat,quat_to_euler,quat_to_mat
from dtsd.envs.src.env_frame_recorder import frame_recorder,frame_recorder_dummy
from dtsd.envs.src.trajectory_class import biped_trajectory_preview
from dtsd.envs.src.env_logger import logger,logger_dummy
from dtsd.envs.sim.mujoco_sim_base import mujoco_sim
from dtsd.envs.src.metric2dist import metric2dist
from dtsd.envs.src.misc_funcs import *
from itertools import product
import dtsd.envs.src.observations as obs
import dtsd.envs.src.rewards as rew
import dtsd.envs.src.actions as act
import numpy as np 
import importlib
import datetime
import torch
import yaml
import os

EXP_DIR_PATH = os.path.dirname(__file__).replace('dtsd/envs/sim','')

class ogpo_biped_base:
  
  def __init__(self, exp_conf_path='./exp_confs/default.yaml'):

    conf_file = open(exp_conf_path) 
    self.exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)

    # for loading files in the remote, update any paths in the config
    self.exp_conf['sim_params']['model_path'] = os.path.join(
                                                              EXP_DIR_PATH,
                                                              self.exp_conf['sim_params']['model_path']
                                                            )
    model_prop_path = self.exp_conf['sim_params']['model_path'].replace('.xml','.yaml')
    prop_file = open(model_prop_path) 
    self.model_prop = yaml.load(prop_file, Loader=yaml.FullLoader)    

    # set task param changes to span discrete_grid and treat it like discrete
    for trk_elem in self.exp_conf['task']['modes'].keys():
      if self.exp_conf['task']['modes'][trk_elem]['param_dist']['type'] == 'discrete_grid':
        
        params_values = []

        for p_i,p_name in enumerate(self.exp_conf['task']['modes'][trk_elem]['param_names']):
          params_values.append(
                                np.linspace(
                                            self.exp_conf['task']['modes'][trk_elem]['param_dist']['support'][0][p_i],
                                            self.exp_conf['task']['modes'][trk_elem]['param_dist']['support'][-1][p_i],
                                            self.exp_conf['task']['modes'][trk_elem]['param_dist']['density'][p_i]
                                          ).tolist()
                              )     
        
        param_values = list(product(*params_values))
        for pvi,pv in enumerate(param_values):
          param_values[pvi] = list(pv)
        self.exp_conf['task']['modes'][trk_elem]['param_dist']['points'] = param_values

    # any default values
    if 'epi_steps_max_thresh' not in self.exp_conf['terminations'].keys():
      self.exp_conf['terminations']['epi_steps_max_thresh'] = np.inf
      print('epi_steps_max_thresh not found in exp_conf, setting to np.inf')

    # set simulator base
    self.sim = mujoco_sim( 
                    **self.exp_conf['sim_params']
                    )


    # set render,vis variants of functions
    if exists_and_true('visualize_oracle',self.exp_conf):
      self.call_oracle = self.call_oracle_w_vis
    else:
      self.call_oracle = self.call_oracle_wo_vis 

    # for counters, buffers, phase and clock
    self.true_base_pos_traj_this_epi = []
    self.oref_base_pos_traj_this_epi = []
    self.episode_counter = 0
    self.this_epi_nstep  = 0 
    self.phase_add = 1
    # simulate X mujoco steps with same pd target, action_repeat
    self.simrate = self.exp_conf['simrate'] 
    self.dt =  self.simrate * self.sim.dt
    self.phase   = 0  

    # initialise oracle
    orac_class_name = self.exp_conf['oracle']['entry'].split('.')[-1]
    orac_file_entry = self.exp_conf['oracle']['entry'].replace('.'+orac_class_name,'')    
    orac_module = importlib.import_module(orac_file_entry)
    self.oracle = getattr(orac_module,orac_class_name)(terrain_map_resolution = [self.sim.hfield_x_index2len,self.sim.hfield_y_index2len]) 

    # based on task descryp set a mode sampler
    self.transition_samplers = []
    for i in range(len(self.exp_conf['task']['modes'].keys())):
      self.transition_samplers.append(
                                      metric2dist(
                                                  n_cases = len(self.exp_conf['task']['modes'].keys()),
                                                  **self.exp_conf['task']
                                                )
                                      )

    # load a dummy traj to prepare observation space
    self.generate_task_variant()
    self.call_oracle()

    # set mode encoder
    if 'mode_encoder' in self.exp_conf.keys():
        self.exp_conf['mode_encoder']['model_path'] = os.path.join(EXP_DIR_PATH,
                                                          self.exp_conf['mode_encoder']['model_path'])
        model = torch.load(self.exp_conf['mode_encoder']['model_path'],map_location=torch.device('cpu'))
        self.mode_enc = model.encoder
        # to save re-computing latent from the ae for obs
        self.curr_mode_latent = None

    # intialise the MDP
    self.prepare_observation_space()
    self.prepare_reward_function()
    act_dim = self.prepare_action_space()
    
    dummy_obs = self.get_full_obs()
    self.observation_space = np.zeros_like(dummy_obs)
    self.action_space  = np.zeros(act_dim)

    # set pd gains as list
    if not isinstance(self.exp_conf['p_gain'],list):
      self.exp_conf['p_gain'] = [self.exp_conf['p_gain']]*act_dim
    
    if not isinstance(self.exp_conf['d_gain'],list):
      self.exp_conf['d_gain'] = [self.exp_conf['d_gain']]*act_dim
    
    # internal buffers and book-keeping
    self.curr_observation = np.zeros_like(dummy_obs)
    self.curr_action = np.zeros(act_dim)
    self.curr_total_reward = 0
    self.prev_action = np.zeros(act_dim)
    self.prev_ctrl = np.zeros_like(self.sim.data.ctrl)
    
    # internal loggers, recorder while testing
    this_exp_date = datetime.datetime.now().strftime("%d%b%Y")
    this_exp_time = datetime.datetime.now().strftime("%H:%M")

    # frame recorder
    if exists_not_none('frame_recorder',self.exp_conf):
      self.exp_conf['frame_recorder']['export_date_time'] = this_exp_date+'/'+this_exp_time
      self.frame_recorder = frame_recorder(
                                            self.exp_conf['frame_recorder']
                                          )
    else:
      self.frame_recorder = frame_recorder_dummy(None)
    
    # logger
    if 'export_logger' in self.exp_conf.keys():
      self.exp_conf['export_logger']['export_date_time'] = this_exp_date+'/'+this_exp_time
      self.export_logger = logger(
                                  logger_conf=self.exp_conf['export_logger'],
                                  )
    
    else:
      self.export_logger = logger_dummy(None)

  def step_simulation(self, action):
      self.prev_ctrl = np.copy(self.sim.data.ctrl)
      ctrl = self.get_ctrl_from_action(action)
      
      # set visual references ate very sim step to keep it alive 
      if self.exp_conf['visualize_reference']:
        r_qpos,_ = self.get_ref_state(self.phase)
        for i, bp_id in enumerate(self.model_prop['vclone']['ids']['base_pos']):
          self.sim.data.qpos[bp_id] = r_qpos[i]
        for i, bp_id in enumerate(self.model_prop['vclone']['ids']['base_ori']):
          self.sim.data.qpos[bp_id] = r_qpos[i+3]
      
      # set control
      self.sim.set_control(ctrl)

      # simulate a step
      self.sim.simulate_n_steps(n_steps=1)
      
      # update internal logger
      self.export_logger.update(self,self.sim.data)

  def step(self, action):
      
      # call preview to generate reference
      if self.this_epi_nstep % self.exp_conf['oracle']['prediction_horizon'] == 0:
      
        if self.exp_conf['visualize_reference']:            
            self.sim.set_vclone_color(set='switch')
        self.call_oracle()

      # apply action
      self.curr_action = np.copy(action)
      for i in range(self.simrate):
          self.step_simulation(action)        

      # record frame
      self.frame_recorder.append_frame(self.sim)
      
      # compute rewards
      info_dict = {}
      if exists_and_true('return_rew_dict',self.exp_conf):
        reward, rew_dict =  self.compute_reward(return_rew_dict=True)
        info_dict.update({'rewards':rew_dict})
      else:
        reward = self.compute_reward()
      
      # early terminations
      done,termin_condn = self.check_terminations(current_reward=reward)
      info_dict['termination_condition'] = termin_condn
      if done:
        export_name = "epi_"+str(self.episode_counter)
        self.frame_recorder.export(export_name = export_name)
        self.export_logger.export(export_name = export_name)

      # update trackers and buffers 
      self.true_base_pos_traj_this_epi.append(self.get_robot_base_pos().tolist())
      self.oref_base_pos_traj_this_epi.append(self.get_ref_state(self.phase)[0][0:3].tolist())
      self.prev_action = np.copy(self.curr_action)
      self.this_epi_nstep  += 1
      self.phase += self.phase_add
      obs = self.get_full_obs()

      return obs, reward, done, info_dict
  
  def reset(
            self
           ):

      self.export_logger.reset()
      self.prev_action = np.zeros_like(self.prev_action)
      self.this_epi_nstep = 0
      self.true_base_pos_traj_this_epi = []
      self.oref_base_pos_traj_this_epi = []
      self.episode_counter += 1

      # reset the frame recorder
      self.frame_recorder.reset()
      
      # reset the sim
      self.sim.reset()

      # constrain the robot
      self.constraint_robot()

      # select task variant      
      self.generate_task_variant()      
      
      # initase the robot
      self.initialize_robot()

      # query the oracle
      self.call_oracle()

      # model properties for domain randomisation
      self.set_model_parameters()

      # set vclone color
      if self.exp_conf['visualize_reference']:            
        self.sim.set_vclone_color(set='green') 
  
      obs0 = self.get_full_obs()
      return obs0

  def close(self):
    self.sim.close()

  def check_terminations(
                          self,
                          current_reward,
                          verbose=False
                        ):
    
    # 'epi_steps_max_thresh' common for all envs
    if self.this_epi_nstep >= self.exp_conf['terminations']['epi_steps_max_thresh']:
      if verbose: print('done','epi_steps_max_thresh')
      return True, 'epi_steps_max_thresh'

    return False, None


  def constraint_robot(self): 
    # activate constraint, world_root constraint assumed to be 0
    self.sim.model.eq_active0[0] = 1    
    self.sim.data.eq_active[0] = 1

    # simulate to settle
    tvel_norm = np.inf
    while tvel_norm > 1e-3:
      self.sim.simulate_n_steps(1)
      base_tvel = self.get_robot_base_tvel()
      tvel_norm = np.linalg.norm(base_tvel)

    # set joint to nominal configuration
    jpn = np.array(self.model_prop[self.exp_conf['robot']]['jpos_nominal']) 
    error_norm = np.inf    
    while error_norm > 1e-1:
      error_norm = 0
      for i,(jci,jpi,jvi) in enumerate(
                                              zip(
                                                  self.model_prop[self.exp_conf['robot']]['ids']['jctrl'],                                                
                                                  self.model_prop[self.exp_conf['robot']]['ids']['jpos'],
                                                  self.model_prop[self.exp_conf['robot']]['ids']['jvel'],
                                                  )
                                          ):
          
          error_jpos = jpn[i] - self.sim.data.qpos[jpi]
          error_jvel = 0 - self.sim.data.qvel[jvi]
          error_norm += error_jpos**2 + error_jvel**2
          self.sim.data.ctrl[jci] =  self.exp_conf['p_gain'][i]*(error_jpos) \
                      + self.exp_conf['d_gain'][i]*(error_jvel)
      error_norm = np.sqrt(error_norm)
      self.sim.simulate_n_steps(1)
      
  def initialize_robot(self):
    
    # deactivate constraint, world_root constraint assumed to be 0
    self.sim.model.eq_active0[0] = 0      
    self.sim.data.eq_active[0] = 0

    # zero our velocity as the ideal initial condition
    self.sim.data.qvel[:] = 0.0    
    
    # base state initialisation
    if 'set_robot_nominal' in self.exp_conf['initialisations'].keys():
      # set robot to nominal height 
      self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] = self.model_prop[self.exp_conf['robot']]['height_nominal']

    if 'set_robot_base_pos_x' in self.exp_conf['initialisations'].keys():
      self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][0]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_pos_x'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_pos_x'][-1] 
                          )
    
    if 'set_robot_base_pos_y' in self.exp_conf['initialisations'].keys():
      self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][1]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_pos_y'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_pos_y'][-1] 
                          )

    if 'set_robot_h_on_terrain' in self.exp_conf['initialisations'].keys():
      base_pos = self.get_robot_base_pos()
      terrain_height = self.sim.get_terrain_height_at(base_pos)
      
      height = terrain_height + np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_h_on_terrain'][0], 
                          high=self.exp_conf['initialisations']['set_robot_h_on_terrain'][-1] 
                          )
      if height >= self.model_prop[self.exp_conf['robot']]['height_nominal']:
        self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] = height
      else:
        print("settling robot at lower height:",height)
        self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] = self.model_prop[self.exp_conf['robot']]['height_nominal']
        while self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] > height:
          self.sim.simulate_n_steps(1)        
    
    
    base_rpy = [0,0,0]
    update_base_rpy = False
    
    if 'set_robot_base_roll' in self.exp_conf['initialisations'].keys():
      update_base_rpy = True
      base_rpy[0] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_roll'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_roll'][-1] 
                          )
    
    if 'set_robot_base_pitch' in self.exp_conf['initialisations'].keys():
      update_base_rpy = True
      base_rpy[1] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_pitch'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_pitch'][-1] 
                          )
    
    if 'set_robot_base_yaw' in self.exp_conf['initialisations'].keys():
      update_base_rpy = True

      base_rpy[2] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_yaw'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_yaw'][-1] 
                          )

    if update_base_rpy:
      self.sim.data.qpos[
                            self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][0]:
                        1 + self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][-1]
                        ] = euler_to_quat(np.radians(base_rpy))
      
    if 'set_robot_flat_on_terrain' in self.exp_conf['initialisations'].keys():
      # conf = self.exp_conf['initialisations']['set_robot_flat_on_terrain']
      # for type in conf['type']:

      #   if type ==  'rand_ori_drop':

          
      #     # pitch perturb
      #     base_rpy[1] = np.random.uniform(
      #                       low=self.exp_conf['initialisations']['set_robot_flat_on_terrain']['set_robot_pitch'][0], 
      #                       high=self.exp_conf['initialisations']['set_robot_flat_on_terrain']['set_robot_pitch'][-1] 
      #                       )
          
      #     # roll perturb
      #     base_rpy[0] = np.random.uniform(
      #                       low=self.exp_conf['initialisations']['set_robot_flat_on_terrain']['set_robot_roll'][0], 
      #                       high=self.exp_conf['initialisations']['set_robot_flat_on_terrain']['set_robot_roll'][-1] 
      #                       )
      #     # print(base_rpy)
      #     self.sim.data.qpos[
      #                           self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][0]:
      #                       1 + self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][-1]
      #                       ] = euler_to_quat(np.radians(base_rpy))
        
      #   if type == 'rand_height_drop':
          
      #     height = np.random.uniform(
      #                       low=self.exp_conf['initialisations']['set_robot_flat_on_terrain']['set_robot_height'][0], 
      #                       high=self.exp_conf['initialisations']['set_robot_flat_on_terrain']['set_robot_height'][-1] 
      #                       )
          
      #     self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] = height

      base_rpy = self.curr_mode['param'][:3]
      base_height = self.curr_mode['param'][3]
      self.sim.data.qpos[
                                self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][0]:
                            1 + self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][-1]
                            ] = euler_to_quat(base_rpy)
      self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] = base_height
      
      
      contact = False
      self.sim.simulate_n_steps(2)
      while not (np.abs(self.get_robot_base_tvel()[-1])<5e-2 and contact):
        # self.sim.viewer.sync()  
        self.sim.simulate_n_steps(1)      
        
        for i in range(self.sim.model.nbody):
          body_name = self.sim.obj_id2name(obj_id=i,type='body')
          if body_name not in ['L_toe','R_toe','l_toe', 'r_toe']:
            contact, _ = self.sim.contact_bw_bodies(body1='terrain',body2=body_name)
            if contact:
                break
        
        

      
      
    if 'set_robot_base_tvel_x' in self.exp_conf['initialisations'].keys():
      self.sim.data.qvel[self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][0]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_tvel_x'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_tvel_x'][-1] 
                          )

    if 'set_robot_base_tvel_y' in self.exp_conf['initialisations'].keys():
      self.sim.data.qvel[self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][1]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_tvel_y'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_tvel_y'][-1] 
                          )

    if 'set_robot_base_tvel_z' in self.exp_conf['initialisations'].keys():
      self.sim.data.qvel[self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][2]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_tvel_z'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_tvel_z'][-1] 
                          )

    if 'set_robot_base_avel_roll' in self.exp_conf['initialisations'].keys():
      self.sim.data.qvel[self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][0]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_avel_roll'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_avel_roll'][-1] 
                          )

    if 'set_robot_base_avel_pitch' in self.exp_conf['initialisations'].keys():
      self.sim.data.qvel[self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][1]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_avel_pitch'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_avel_pitch'][-1] 
                          )

    if 'set_robot_base_avel_yaw' in self.exp_conf['initialisations'].keys():
      self.sim.data.qvel[self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][2]] = np.random.uniform(
                          low=self.exp_conf['initialisations']['set_robot_base_avel_yaw'][0], 
                          high=self.exp_conf['initialisations']['set_robot_base_avel_yaw'][-1] 
                          )

    # joint/genealized state initialisation
    for key in self.exp_conf['initialisations'].keys():
      if 'set_robot_qpos_' in key:
        i = int(key.replace('set_robot_qpos_',''))
        self.sim.data.qpos[i] = np.random.uniform(
                            low=self.exp_conf['initialisations'][key][0], 
                            high=self.exp_conf['initialisations'][key][-1] 
                            )

    # one step to settle
    self.sim.simulate_n_steps(1)

  def set_model_parameters(self):
    
    if exists_not_none ('domain_randomisation',self.exp_conf):
      

      for prop in self.exp_conf['domain_randomisation'].keys():

        if 'body' in prop:
          for body_id in range(self.sim.model.nbody):          
            body_name = self.sim.obj_id2name(obj_id=body_id,type='body')

            if 'v_' not in body_name:
              f_nom = 1
              if self.exp_conf['domain_randomisation'][prop]['scheme'] == 'uniform_random':
                f_nom = np.random.uniform(
                                            
                                                low = self.exp_conf['domain_randomisation'][prop]['range'][0], 
                                                high =self.exp_conf['domain_randomisation'][prop]['range'][-1] 
                                            
                                            )
              getattr(self.sim, 'set_'+prop)(body=body_name, fraction_nominal=f_nom)
        else:
          f_nom = 1
          if self.exp_conf['domain_randomisation'][prop]['scheme'] == 'uniform_random':
            f_nom = np.random.uniform(
                                        
                                            low = self.exp_conf['domain_randomisation'][prop]['range'][0], 
                                            high =self.exp_conf['domain_randomisation'][prop]['range'][-1] 
            )

          getattr(self.sim, 'set_'+prop)(fraction_nominal=f_nom)

  def call_oracle_w_vis(self):

    # to clear all
    self.sim.clear_user_scn() 

    if self.true_base_pos_traj_this_epi:
      # visualise the previous horizon
      self.sim.put_connector(
                              points=self.true_base_pos_traj_this_epi,
                              rgba=[1.0,0.0,0.0,1.0],
                            )
      self.sim.put_connector(
                              points=self.oref_base_pos_traj_this_epi,
                              rgba=[0.0,1.0,0.0,1.0],
                            )

    # generate reference
    self.call_oracle_wo_vis()

    # clean buffers
    self.true_base_pos_traj_this_epi = []
    self.oref_base_pos_traj_this_epi = []

    # visualise the current horizon
    self.sim.put_connector(points=self.curr_ref_traj.qpos[:self.exp_conf['oracle']['prediction_horizon'],0:3])
    
    # for li_orac in dive task
    # modified_ref = self.curr_ref_traj.qpos[:self.exp_conf['oracle']['prediction_horizon'],0:3].copy()
    # modified_ref[:,1] = -1*modified_ref[:,1]
    # self.sim.put_connector(
    #                         points=modified_ref,
    #                         )

  def call_oracle_wo_vis(self):
    self.generate_reference()
    self.phase = 0  
    self.phaselen = len(self.curr_ref_traj)

  def generate_reference(self):      
      raise NotImplementedError

  def generate_task_variant(self):
    raise NotImplementedError

  def get_ref_state(self,phase=None):
      if phase is None:
          phase = self.phase
      pos = np.copy(self.curr_ref_traj.qpos[phase])
      vel = np.copy(self.curr_ref_traj.qvel[phase])
      return pos, vel

  def prepare_reward_function(self):
      """ Prepares a list of reward functions, whcih will be called to compute the total reward.
          Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
      """
      # remove zero scales + multiply non-zero ones by dt
      for key in list(self.exp_conf['rewards']['scales'].keys()):
          scale = self.exp_conf['rewards']['scales'][key]
          if scale == 0:
              self.exp_conf['rewards']['scales'].pop(key)
          # else:
          #     self.exp_conf['rewards']['scales'][key] *= self.dt
      
      # prepare list of functions
      self.reward_functions = []
      self.reward_names = []
      for name, scale in self.exp_conf['rewards']['scales'].items():

          self.reward_names.append(name)
          # name = '_reward_' + name
          self.reward_functions.append(getattr(rew, name))

  def prepare_observation_space(self):
      """ Prepares a list of observations, which will be concatenated to compute the total observation.
      """
      # prepare list of functions
      self.observation_functions = []
      self.observation_names = []
      for name in self.exp_conf['observations'].keys():
          self.observation_names.append(name)
          self.observation_functions.append(getattr(obs, name))

  def prepare_action_space(self):
      """ Prepares a list of actions, which will be concatenated to compute the total action.
      """
      # prepare list of functions
      self.action_functions = []
      self.action_names = []
      max_act_id = 0
      for name in self.exp_conf['actions'].keys():
          self.action_names.append(name)
          if max_act_id < self.exp_conf['actions'][name][-1]:
            max_act_id = self.exp_conf['actions'][name][-1]
          self.action_functions.append(getattr(act, name))
      return max_act_id

  def get_full_obs(self):
    # NOTE: the order matters for not breaking the old code
      obs = []
      for i in range(len(self.observation_functions)):
          name = self.observation_names[i]
          obs.append(self.observation_functions[i](self))
      
      obs = np.concatenate(obs).ravel()
      self.curr_observation = np.copy(obs)
      return obs

  def get_ctrl_from_action(self,action):
    # NOTE: the order matters for not breaking the old code
      ctrl = np.zeros_like(self.sim.data.ctrl)
      for i in range(len(self.action_functions)):
          name = self.action_names[i]
          action_ids = self.exp_conf['actions'][name]
          action_subset = action[action_ids[0]:action_ids[-1]]
          ctrl += self.action_functions[i](self,action_subset) 
      
      return ctrl
  
  def compute_reward(self,return_rew_dict=False):
      """ Compute rewards
          Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
          adds each terms to the episode sums and to the total reward
      """
      total_reward = 0.

      if return_rew_dict:
        rew_dict = {}

      for i in range(len(self.reward_functions)):
          name = self.reward_names[i]
          rew = self.reward_functions[i](self) * self.exp_conf['rewards']['scales'][name]
          if return_rew_dict:
            rew_dict.update({name:np.round(rew,2)})
          total_reward += rew
      

      self.curr_total_reward = total_reward
      if return_rew_dict:
        rew_dict.update({'sum':np.round(total_reward,2)})
        return total_reward,rew_dict
      else:
        return total_reward
  
  def get_robot_base_pos(self):
    return self.sim.data.qpos[
                              self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][0]:
                              1+self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]          
                            ]

  def get_robot_base_ori(self,format='quat'):
    base_ori = self.sim.data.qpos[
                              self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][0]:
                              1+self.model_prop[self.exp_conf['robot']]['ids']['base_ori'][-1]          
                            ]
    if format == 'rpy':
      base_ori = quat_to_euler(base_ori)
    elif format == 'rotmat':
      base_ori = quat_to_mat(base_ori)[0:3,0:3]  
    return base_ori

  def get_robot_base_tvel(self):
    return self.sim.data.qvel[
                              self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][0]:
                              1+self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][-1]          
                            ]

  def get_robot_base_avel(self):
    return self.sim.data.qvel[
                              self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][0]:
                              1+self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][-1]          
                            ]
  
  def get_robot_base_tacc(self):
    return self.sim.data.qacc[
                              self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][0]:
                              1+self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][-1]          
                            ]

  def get_robot_base_aacc(self):
    return self.sim.data.qacc[
                              self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][0]:
                              1+self.model_prop[self.exp_conf['robot']]['ids']['base_avel'][-1]          
                            ]

  def get_robot_base_state(self):
    base_pose = np.concatenate([self.get_robot_base_pos(),self.get_robot_base_ori()])
    base_vel = np.concatenate([self.get_robot_base_tvel(),self.get_robot_base_avel()])
    return np.concatenate([base_pose,base_vel])

  def get_mode_latent_from_traj(self,traj):
    if not isinstance(traj,torch.Tensor):
      traj = torch.from_numpy(traj)
    try:  
      # if input is valid, which is always the case (used as is faster than if else)
      mode_latent = self.mode_enc(traj)
      return mode_latent.detach().numpy()
    except:
      return None
  
  # place holder for now
  def log_stuff(self,iteration): 
    pass
  
  def makedirs_for_log_stuff(self):
    pass