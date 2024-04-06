import torch
import hashlib
import os
import numpy as np
from collections import OrderedDict
import yaml
import time
from tqdm.auto import tqdm, trange
import multiprocessing
import copy
from itertools import product
from dtsd.envs.src.misc_funcs import *
from src.misc_funcs import make_time_series_animation
from dtsd.envs.src import transformations
# tmp for testing
import mediapy as media


# for fbg infinite
'''
class camera_trolly:
	def __init__(self):
		# self.pos = [5.0,0,0.5] 
		# set_robot_base_pos_z: [0.8]

		# self.pos = [0.5,0,0.5] # lms_test_x0
		self.pos = [-9.0,0,0.55] 
		
		self.azim = 0 #90
		self.elev = -7
		self.dist = 2.0
		# self.pos = [0.0,0,0.5]
	
		# self.delta_x = 0.0 # lms_test_x0
		self.delta_x =  0.8*0.03 #0.3*0.03

		self.delta_azim = 0.62
		self.delta_elev = 0.0
		self.delta_dist = 0.01
		
		self.sync_subj_pos_thresh = 0.1 #None
	def update(self, subject_pos=None):
		
		
		# self.pos[0] += self.delta_x
		self.pos[0] = subject_pos[0] 
		self.elev += self.delta_elev


		if subject_pos is not None:
			
			# if self.sync_subj_pos_thresh is not None:
			# 	if np.abs(self.pos[0] - subject_pos[0]) > self.sync_subj_pos_thresh:
			# 		self.pos[0] = subject_pos[0]

	
			if subject_pos[0] < -5.0:
				# clip dist
				self.dist += self.delta_dist
				# self.azim += self.delta_azim
				# self.azim = np.clip(self.azim,0,90)
				self.azim -= self.delta_azim
				self.azim = np.clip(self.azim,-90,0)


			elif subject_pos[0] > 5.0:
				self.dist -= self.delta_dist
				# self.azim += self.delta_azim
				# self.azim = np.clip(self.azim,90,180)
				self.azim -= self.delta_azim
				self.azim = np.clip(self.azim,-180,-90)
			self.dist = np.clip(self.dist,2.0,4.5)
'''


class camera_trolly:
	def __init__(self):
		# self.pos = [5.0,0,0.5] 
		# set_robot_base_pos_z: [0.8]

		# self.pos = [0.5,0,0.5] # lms_test_x0
		# self.pos = [2.5,0,0.5] # lms_test_x0
		self.pos = np.array([0.0,0,0.0])

		# self.pos = [-8.25,0,0.55] 
		# self.pos = [-7.5,0,0.55] 
		
		self.azim = 135 #90
		self.elev = -20 #-7
		self.dist = 0.75 #4.0 #3.0 #4.0 #3.25
		# self.pos = [0.0,0,0.5]
	
		self.delta_pos = np.array([0.1,0.0,0.0])
		# self.delta_azim = 0.62
		# self.delta_elev = 0.0
		self.delta_dist = 0.105
		
	
	def update(self, subject_pos=None):
		
		# if subject_pos[0]  > 0.6:
		# 	self.pos+= self.delta_pos
		# 	self.pos[0] = 5 if self.pos[0] > 5.0 else self.pos[0]
		# 	self.dist += self.delta_dist
		# 	self.dist = 7.25 if self.dist > 7.25 else self.dist
		pass

# for dive 
'''
class camera_trolly:
	def __init__(self):

		
		flip_type = 'right' 
		self.dist = 3.0 #4.75#4.0

		self.elev = -5 #-10


		if flip_type in ['front','back']:
			self.pos = [0.5,0.0,1.45] #[0.0,0.0,0.75] 
			self.azim = 90
		
		elif flip_type in ['left','right']:
			# self.pos = [0.0,-0.5,1.45] #[0.0,0.0,0.75] 
			self.pos = [0.0,0.0,0.75] #[0.0,0.0,0.75] 

			self.azim = 180
		
		else:
			self.pos = [0.0,0.0,0.75] 
			self.azim = 135



		self.delta_elev = 0.0
		self.delta_dist = 0.01
		
		self.sync_subj_pos_thresh = 0.1 #None
	def update(self, subject_pos=None):

		# self.pos[2] = subject_pos[2] 
		# self.azim += self.delta_azim
		pass
'''
# for mpst
'''
class camera_trolly:
	def __init__(self):

		self.dist = 4.25#1.5
		self.elev = -5
		self.azim = 180 #90
		self.delta_elev = 0.0
		self.delta_dist = 0.0
		self.pos = [0.0,0.5,1.50] #[0.25,0.0,0.5]
		
	def update(self, subject_pos=None):

		# self.pos[2] = subject_pos[2] 
		# self.azim += self.delta_azim
		pass
'''
def load_lme_from_all_logs(path2logs):
	loglist = os.listdir(path2logs)
	loglist = [file for file in loglist if '.npz' in file]
	lme = None
	for i,log_name in enumerate(loglist):
		worker_log = np.load(os.path.join(path2logs,log_name),allow_pickle=True)
		if lme is None:
			lme = worker_log['mode_latents_encountered']
		else:
			lme = np.vstack((lme,worker_log['mode_latents_encountered']))
	return lme

def create_logger(args):
	from torch.utils.tensorboard import SummaryWriter
	"""Use hyperparms to set a directory to output diagnostic files."""

	arg_dict = args.__dict__
	assert "logdir" in arg_dict, \
		"You must provide a 'logdir' key in your command line arguments."

	# sort the keys so the same hyperparameters will always have the same hash
	arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

	# remove seed so it doesn't get hashed, store value for filename
	# same for logging directory
	if 'seed' in arg_dict:
		seed = str(arg_dict.pop("seed"))
	else:
		seed = None
	
	logdir = str(arg_dict.pop('logdir'))

	# get a unique hash for the hyperparameter settings, truncated at 10 chars
	if seed is None:
		arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6]
	else:
		arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed

	# output_dir = os.path.join(logdir, arg_hash)
	conf_name = args.exp_conf_path.split('/')[-1]
	output_dir = os.path.join(logdir, conf_name.replace('.yaml',''))

	# create a directory with the hyperparm hash as its name, if it doesn't
	# already exist.
	os.makedirs(output_dir, exist_ok=True)

	# Create a file with all the hyperparam settings in plaintext
	info_path = os.path.join(output_dir, "experiment.info")
	file = open(info_path, 'w')
	for key, val in arg_dict.items():
			file.write("%s: %s" % (key, val))
			file.write('\n')

	# copy the exp_conf_file
	default_env_conf_path = './exp_confs/default.yaml'

	default_conf_file = open(default_env_conf_path)
	default_exp_conf = yaml.load(default_conf_file, Loader=yaml.FullLoader)

	given_conf_file = open(args.exp_conf_path) # remove
	given_exp_conf = yaml.load(given_conf_file, Loader=yaml.FullLoader)

	merged_exp_conf = default_exp_conf
	for key in given_exp_conf.keys():


			if key in merged_exp_conf.keys():
					merged_exp_conf[key] = given_exp_conf[key]
			else:
					merged_exp_conf.update({key:given_exp_conf[key]})

	args.exp_conf_path = os.path.join(output_dir,'exp_conf.yaml')
	final_conf_file =  open(args.exp_conf_path,'w')
	yaml.dump(merged_exp_conf,final_conf_file,default_flow_style=False,sort_keys=False)

	logger = SummaryWriter(output_dir, flush_secs=0.1)
	logger.dir = output_dir

	logger.arg_hash = arg_hash
	return logger

def train_normalizer(policy,
										 min_timesteps, 
										 max_traj_len=1000, 
										 noise=0.5,
										 exp_conf_path="./exp_confs/default.yaml"
										 ):
	with torch.no_grad():
		env = env_factory(exp_conf_path)()
		env.dynamics_randomization = False
		total_t = 0
		pbar = tqdm(total=min_timesteps, desc="Training normalizer", leave=False)
		while total_t < min_timesteps:
			state = env.reset()
			done = False
			timesteps = 0

			if hasattr(policy, 'init_hidden_state'):
				policy.init_hidden_state()

			prev_total_t = total_t
			while not done and timesteps < max_traj_len:
				action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
				state, _, done, _ = env.step(action)
				timesteps += 1
				total_t += 1
			
			pbar.update(total_t-prev_total_t)

def env_factory(
								# dynamics_randomization,
								exp_conf_path,
								):
		from functools import partial


		conf_file = open(exp_conf_path)
		exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)

		"""
		Returns an *uninstantiated* environment constructor.
		Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
		this allows us to pass their constructors to Ray remote functions instead 
		"""
		
		
		import importlib

		if 'env_entry' in exp_conf.keys():
			
			env_class_name = exp_conf['env_entry'].split('.')[-1]
			# env_file_entry = exp_conf['env_entry'].replace('.'+env_class_name,'')
			env_file_entry = '.'.join(exp_conf['env_entry'].split('.')[:-1])
			env_module = importlib.import_module(env_file_entry)
			biped_env = getattr(env_module,env_class_name) 
		else:
			biped_env = importlib.import_module('dtsd.envs.biped_parkour').biped_env

		return partial(
										biped_env, 
										exp_conf_path = exp_conf_path,
										)

def eval_policy_trng(model, 
								env=None, 
								episodes=5, 
								max_traj_len=400, 
								verbose=True, 
								exp_conf_path = './exp_confs/default.yaml',
								iteration = 0,
								):
	if env is None:
		env = env_factory(exp_conf_path)()

	if model.nn_type == 'policy':
		policy = model
	elif model.nn_type == 'extractor':
		policy = torch.load(model.policy_path)

	with torch.no_grad():
		steps = 0
		ep_returns = []
		
		for _ in range(episodes):
			env.dynamics_randomization = False
			
			state = torch.Tensor(env.reset())
			done = False
			traj_len = 0
			ep_return = 0

			if hasattr(policy, 'init_hidden_state'):
				policy.init_hidden_state()
			
			while not done and traj_len < max_traj_len:

				action = policy(state)
				next_state, reward, done, _ = env.step(action.numpy())

				state = torch.Tensor(next_state)

				ep_return += reward
				traj_len += 1
				steps += 1

				if model.nn_type == 'extractor':
					pass

			ep_returns += [ep_return]
			if verbose:
				print('Return: {:6.2f}'.format(ep_return))


	# any logging after evaluation:
	if iteration == 0:
		env.makedirs_for_log_stuff()
	env.log_stuff(iteration)

	return np.mean(ep_returns)

def mode_parameter_test_worker(
								anlys_log_path,
								tstng_exp_confs,
								policy_path,
								worker_id,
								prev_last_idx,
							):

	total = len(tstng_exp_confs)
	text = "# Worker {0}".format(worker_id)
	with torch.no_grad():
		for _,tstng_exp_conf in zip(trange(total, 
											desc=text, 
											disable= False,
											leave=False,
											lock_args=None ,
											position=worker_id
											), # proper printing, placing settings for the pbars

										tstng_exp_confs):

			tstng_exp_conf_path = anlys_log_path + str(prev_last_idx+_)+'.yaml'
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')

			yaml.dump(
						tstng_exp_conf,
						tstng_exp_conf_file,
						default_flow_style=False,
						sort_keys=False
						)
			tstng_exp_conf_file.close()

			# update export paths
			# tstng_exp_conf_path['export_logger']['export_path'] = anlys_log_path


			# load policy to env
			policy = torch.load(os.path.join(policy_path,'actor.pt'))
			if hasattr(policy, 'init_hidden_state'):
				policy.init_hidden_state()

			env = env_factory(tstng_exp_conf_path)()
			if 'frame_recorder'in env.exp_conf.keys():
				env.exp_conf['frame_recorder']['export_path'] = anlys_log_path + str(prev_last_idx+_)+'.mp4'
				env.exp_conf['frame_recorder']['avoid_logfolder_names'] = True

			is_render = exists_and_true('render', env.sim.sim_params)
			if exists_not_none('frame_recorder',env.exp_conf):
				is_record = True
			else:
				is_record = False					


			# reset	
			state = torch.Tensor(env.reset())

			# set camera trolly
			cam_trolly = camera_trolly()

			# set viewer for onscreen
			viewer_alive = True
			if is_render:
				env.sim.update_camera(cam_name='viewer')
				env.sim.viewer.update_hfield(0)
			else:
				env.sim.viewer_paused = False
			
			# set renderer for offscreen
			if is_record:
				env.sim.init_renderers()
			done = False
			steps = 0
			returns = 0

			while not done:
				if not env.sim.viewer_paused and viewer_alive:
					action = policy(state)
					next_state, reward, done, info_dict = env.step(action.numpy())

					# set custom camera pos for movable cameras
					cam_trolly.update()

					# onscreen rendering
					if is_render:
						time.sleep(0.02) # for mesh model
						env.sim.update_camera(cam_name='viewer' ,
											pos=cam_trolly.pos,
											azim = cam_trolly.azim,
											elev = cam_trolly.elev,
											dist = cam_trolly.dist,		)
						env.sim.viewer.sync()


					# offscreen rendering
					if is_record:
						env.sim.update_camera(cam_name='free_camera',
											pos=cam_trolly.pos,
											azim = cam_trolly.azim,
											elev = cam_trolly.elev,
											dist = cam_trolly.dist,							
												)

					state = torch.Tensor(next_state)
					steps += 1
					returns += reward

				if is_render:
					viewer_alive = env.sim.viewer.is_running()
					if not viewer_alive:
						break



			param_val = list(tstng_exp_conf['task']['track_elements'].values())[0]['param_dist']['points'][0]



			if not viewer_alive:
				break
			
			if is_record:				
				env.sim.delete_renderers()				
			
			# if is_record:
			# 	media.write_video(
			# 						os.path.join(anlys_log_path,str(prev_last_idx+_))+'.mp4',
			# 						frames ,
			# 						fps=int(1/env.dt),
			# 						codec= 'hevc',
			# 					)
			# 	env.sim.delete_renderers()

			np.savez(
					os.path.join(anlys_log_path,str(prev_last_idx+_)),
					epi_len = np.copy(steps),
					returns = np.copy(returns),
					param_val = np.copy(param_val),
					)
			env.close()

def lms_test_x0_worker(
								anlys_log_path,
								mode_latents,
								base_tstng_exp_conf,
								policy_path,
								worker_id,
								prev_last_idx,
							):

	total = len(mode_latents)
	text = "# Worker {0}".format(worker_id)

	with torch.no_grad():
		for _,mode_latent in zip(trange(total, 
											desc=text, 
											disable= False,
											leave=False,
											lock_args=None ,
											position=worker_id
											), # proper printing, placing settings for the pbars

										mode_latents):

			
			
			this_test_log_path = anlys_log_path + str(prev_last_idx+_) 
			os.makedirs(this_test_log_path,exist_ok=True)

			# update export path
			base_tstng_exp_conf['export_logger']['export_path'] = this_test_log_path
			if 'frame_recorder'in base_tstng_exp_conf.keys():
				base_tstng_exp_conf['frame_recorder']['export_path'] = this_test_log_path
				base_tstng_exp_conf['frame_recorder']['avoid_logfolder_names'] = True

			# export the conf
			tstng_exp_conf_path = this_test_log_path+'/conf.yaml'
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
			yaml.dump(
						base_tstng_exp_conf,
						tstng_exp_conf_file,
						default_flow_style=False,
						sort_keys=False
						)
			tstng_exp_conf_file.close()
			
			# load policy to env
			policy = torch.load(policy_path)
			if hasattr(policy, 'init_hidden_state'):
				policy.init_hidden_state()				
			env = env_factory(tstng_exp_conf_path)()
			is_render = exists_and_true('render', env.sim.sim_params)

			if exists_not_none('frame_recorder',env.exp_conf):
				is_record = True
			else:
				is_record = False
			
			# reset	
			state = torch.Tensor(env.reset())

			# set camera trolly
			cam_trolly = camera_trolly()

			# set viewer for onscreen
			viewer_alive = True
			if is_render:
				env.sim.update_camera(cam_name='viewer')
				env.sim.viewer.update_hfield(0)
			else:
				env.sim.viewer_paused = False
			
			# set renderer for offscreen
			if is_record:
				env.sim.init_renderers()
				
			# episode counters
			done = False
			steps = 0
			returns = 0
			# set the mode latetnt 2 be tested
			env.curr_mode_latent = mode_latent
			while not done:
				
				if not env.sim.viewer_paused and viewer_alive:
					action = policy(state)
					next_state, reward, done, info_dict = env.step(action.numpy())  
					
					# set custom camera pos for movable cameras
					cam_trolly.update()
					
					# onscreen rendering
					if is_render:
						time.sleep(0.02) # for mesh model
						env.sim.update_camera(cam_name='viewer' ,pos=cam_trolly.pos)
						env.sim.viewer.sync()

					# offscreen rendering
					if is_record:
						env.sim.update_camera(cam_name='free_camera' ,pos=cam_trolly.pos)

					state = torch.Tensor(next_state)
					steps += 1
					returns += reward

				if is_render:
					viewer_alive = env.sim.viewer.is_running()
					if not viewer_alive:
						break

			if not viewer_alive:
				break

			if is_record:				
				env.sim.delete_renderers()			
			
			np.savez(
					os.path.join(this_test_log_path,'metrics_log.npz'),
					epi_len = np.copy(steps),
					returns = np.copy(returns),
					mode_latent = np.copy(mode_latent),
					)
			env.close()

def n_rollouts_test_worker(
					anlys_log_path,
					policy_path,
					n_rollouts,
					tstng_exp_conf_path,
					worker_id,
					):
		
		np.random.seed(worker_id)
		env = env_factory(tstng_exp_conf_path)()
		is_render = exists_and_true('render', env.sim.sim_params)
		is_record = exists_and_true('record', env.sim.sim_params)
		policy = torch.load(policy_path)
		if hasattr(policy, 'init_hidden_state'):
			policy.init_hidden_state()	
		iteration_times = []

		# fgb
		base_poss_i = []
		base_poss_f = []
		base_rpys_i = []
		base_rpys_f = []

		# common
		episode_length = []
		undisc_returns = []

		# for agility
		base_tvel_means = []
		base_tvel_maxs = []
		base_avel_means = []
		base_avel_maxs = []
		base_tacc_means = []
		base_tacc_maxs = [] 
		base_aacc_means = []
		base_aacc_maxs = []

		mode_latents_encountered = []
		text = "# Worker {0}".format(worker_id) 
		
		with torch.no_grad():
			with trange(
							n_rollouts, 
							desc=text, 
							disable= False,
							leave=False,
							lock_args=None ,
							position=worker_id
						) as pbar, torch.no_grad():
				
				prev_time = 0.0
				
				for n_epi in pbar:
					time_elapsed = pbar.format_dict['elapsed']
					# reset	
					state = torch.Tensor(env.reset())

					# set camera trolly
					cam_trolly = camera_trolly()

					# set viewer for onscreen
					viewer_alive = True
					if is_render:
						env.sim.update_camera(cam_name='viewer')
						env.sim.viewer.update_hfield(0)
					else:
						env.sim.viewer_paused = False
					
					# set renderer for offscreen
					if is_record:
						env.sim.init_renderers()
						frames = []				
						
					# episode counters
					done = False
					steps = 0
					returns = 0

					base_tvels = []
					base_avels = []
					base_taccs = []
					base_aaccs = []

					base_pos_i = env.get_robot_base_pos().copy()
					base_rpy_i = env.get_robot_base_ori(format='rpy').copy()

					while not done:
						if not env.sim.viewer_paused and viewer_alive:

							action = policy(state)
							next_state, reward, done, info_dict = env.step(action.numpy()) 
						
							# set custom camera pos for movable cameras
							cam_trolly.update()
							
							# onscreen rendering
							if is_render:
								# time.sleep(0.02) # for mesh model
								env.sim.update_camera(cam_name='viewer' ,pos=cam_trolly.pos)
								env.sim.viewer.sync()

							# offscreen rendering
							if is_record:
								env.sim.update_camera(cam_name='free_camera' ,pos=cam_trolly.pos)
								pixels = env.sim.get_frame_from_renderer(cam_name='free_camera')
								frames.append(pixels)

							# logs for analysis
							if False in np.isin(env.curr_mode_latent,mode_latents_encountered):
								mode_latents_encountered.append(env.curr_mode_latent)

							# get data
							base_tvel = env.get_robot_base_tvel()
							base_avel = env.get_robot_base_avel()
							base_tacc = env.get_robot_base_tacc()
							base_aacc = env.get_robot_base_aacc()
							# append data
							base_tvels.append(base_tvel)
							base_avels.append(base_avel)
							base_taccs.append(base_tacc)
							base_aaccs.append(base_aacc)


							state = torch.Tensor(next_state)
							steps += 1
							returns += reward
						
						if is_render:
							viewer_alive = env.sim.viewer.is_running()
							if not viewer_alive:
								break


					base_pos_f = env.get_robot_base_pos().copy()
					base_rpy_f = env.get_robot_base_ori(format='rpy').copy()

					# log performance metrics
					base_poss_i.append(base_pos_i)
					base_poss_f.append(base_pos_f)
					base_rpys_i.append(base_rpy_i)
					base_rpys_f.append(base_rpy_f)
					episode_length.append(steps)
					undisc_returns.append(returns)
					base_tvel_means.append(np.mean(base_tvels))
					base_tvel_maxs.append(np.max(base_tvels))
					base_avel_means.append(np.mean(base_avels))
					base_avel_maxs.append(np.max(base_avels))
					base_tacc_means.append(np.mean(base_taccs))
					base_tacc_maxs.append(np.max(base_taccs))
					base_aacc_means.append(np.mean(base_aaccs))
					base_aacc_maxs.append(np.max(base_aaccs))
					iteration_times.append(time_elapsed - prev_time)
					prev_time = time_elapsed
					
					if not viewer_alive:
						break
					if is_record:
						media.write_video(
											os.path.join(anlys_log_path,'worker_log_'+str(worker_id))+'_'+str(n_epi)+'.mp4',
											frames ,
											fps=int(1/env.dt),
											codec= 'hevc',
										)
						env.sim.delete_renderers()
				
		np.savez_compressed(
								os.path.join(anlys_log_path,'worker_log_'+str(worker_id)),
								# all metrics
								episode_length = np.copy(episode_length),
								undisc_returns = np.copy(undisc_returns),
								iteration_times = np.copy(iteration_times),
								mode_latents_encountered = np.copy(mode_latents_encountered),

								base_poss_i = np.copy(base_poss_i),
								base_poss_f = np.copy(base_poss_f),
								base_rpys_i = np.copy(base_rpys_i),
								base_rpys_f = np.copy(base_rpys_f),

								base_tvel_maxs = np.copy(base_tvel_maxs),
								base_avel_maxs = np.copy(base_avel_maxs),
								base_tacc_maxs = np.copy(base_tacc_maxs),
								base_aacc_maxs = np.copy(base_aacc_maxs),

								base_tvel_means = np.copy(base_tvel_means),
								base_avel_means = np.copy(base_avel_means),
								base_tacc_means = np.copy(base_tacc_means),
								base_aacc_means = np.copy(base_aacc_means),


							)
		env.close()

def collect_rollouts_worker(
					anlys_log_path,
					policy_path,
					n_rollouts,
					tstng_exp_conf_path,
					worker_id,
					):
		
		np.random.seed(worker_id)
		env = env_factory(tstng_exp_conf_path)()
		is_render = exists_and_true('render', env.sim.sim_params)
		is_record = exists_and_true('record', env.sim.sim_params)
		policy = torch.load(policy_path)
		if hasattr(policy, 'init_hidden_state'):
			policy.init_hidden_state()	

		# list os trajs
		observations = []
		actions = []
		qposs = []
		qvels = []
		toe_contact_states = []
		
		text = "# Worker {0}".format(worker_id) 
		
		with torch.no_grad():
			with trange(
							n_rollouts, 
							desc=text, 
							disable= False,
							leave=False,
							lock_args=None ,
							position=worker_id
						) as pbar, torch.no_grad():
				
				prev_time = 0.0
				
				for n_epi in pbar:
					time_elapsed = pbar.format_dict['elapsed']
					# reset	
					state = torch.Tensor(env.reset())

					# set camera trolly
					cam_trolly = camera_trolly()

					# set viewer for onscreen
					viewer_alive = True
					if is_render:
						env.sim.update_camera(cam_name='viewer')
						env.sim.viewer.update_hfield(0)
					else:
						env.sim.viewer_paused = False
					
					# set renderer for offscreen
					if is_record:
						env.sim.init_renderers()
						frames = []				
						
					# episode counters
					done = False
					steps = 0
					returns = 0
					obs_traj = []
					act_traj= []
					qpos_traj = []
					qvel_traj = []
					tcs_traj = []
					while not done:
						if not env.sim.viewer_paused and viewer_alive:

							
							obs_traj.append(state.numpy())
							action = policy(state)
							act_traj.append(action.numpy())
							next_state, reward, done, info_dict = env.step(action.numpy()) 
						
							# next state
							qpos = env.sim.data.qpos.copy()
							qvel = env.sim.data.qvel.copy()
							toe_contact_state = env.get_toe_contact_state()
							
							
							qpos_traj.append(qpos)
							qvel_traj.append(qvel)
							tcs_traj.append(toe_contact_state)

							# set custom camera pos for movable cameras
							cam_trolly.update()
							
							# onscreen rendering
							if is_render:
								# time.sleep(0.02) # for mesh model
								env.sim.update_camera(cam_name='viewer' ,pos=cam_trolly.pos)
								env.sim.viewer.sync()

							# offscreen rendering
							if is_record:
								env.sim.update_camera(cam_name='free_camera' ,pos=cam_trolly.pos)
								pixels = env.sim.get_frame_from_renderer(cam_name='free_camera')
								frames.append(pixels)

							state = torch.Tensor(next_state)
							steps += 1
							returns += reward
						
						if is_render:
							viewer_alive = env.sim.viewer.is_running()
							if not viewer_alive:
								break					
					# append trajs
					observations.append(obs_traj)
					actions.append(act_traj)
					qposs.append(qpos_traj)
					qvels.append(qvel_traj)
					toe_contact_states.append(tcs_traj)

					if not viewer_alive:
						break
					if is_record:
						media.write_video(
											os.path.join(anlys_log_path,'worker_log_'+str(worker_id))+'_'+str(n_epi)+'.mp4',
											frames ,
											fps=int(1/env.dt),
											codec= 'hevc',
										)
						env.sim.delete_renderers()
				
		np.savez_compressed(
								os.path.join(anlys_log_path,'worker_log_'+str(worker_id)),
								# all metrics

								observations = np.copy(observations),
								actions = np.copy(actions),
								qposs = np.copy(qposs),
								qvels = np.copy(qvels),
								toe_contact_states = np.copy(toe_contact_states),

							)
		env.close()

def lms_test_tr_worker(
						anlys_log_path,
						mode_latents,
						base_tstng_exp_conf,
						policy_path,
						worker_id,
						prev_last_idx,
					):


	if mode_latents is None:
		mode_latents = [None]
	
	total = len(mode_latents)
	text = "# Worker {0}".format(worker_id)

	with torch.no_grad():
		for _,mode_latent in zip(trange(total, 
											desc=text, 
											disable= False,
											leave=False,
											lock_args=None ,
											position=worker_id
											), # proper printing, placing settings for the pbars

										mode_latents):

			
			if isinstance(prev_last_idx,int):
				this_test_log_path = anlys_log_path + str(prev_last_idx+_)
			else:
				this_test_log_path = anlys_log_path + prev_last_idx
			os.makedirs(this_test_log_path,exist_ok=True)

			# update export path
			base_tstng_exp_conf['export_logger']['export_path'] = this_test_log_path
			if 'frame_recorder'in base_tstng_exp_conf.keys():
				base_tstng_exp_conf['frame_recorder']['export_path'] = this_test_log_path
				base_tstng_exp_conf['frame_recorder']['avoid_logfolder_names'] = True

			# export the conf
			tstng_exp_conf_path = this_test_log_path+'/conf.yaml'
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
			yaml.dump(
						base_tstng_exp_conf,
						tstng_exp_conf_file,
						default_flow_style=False,
						sort_keys=False
						)
			tstng_exp_conf_file.close()
			
			# load policy to env
			policy = torch.load(policy_path)
			if hasattr(policy, 'init_hidden_state'):
				policy.init_hidden_state()				
			env = env_factory(tstng_exp_conf_path)()
			is_render = exists_and_true('render', env.sim.sim_params)

			if exists_not_none('frame_recorder',env.exp_conf):
				is_record = True
			else:
				is_record = False
			
			# reset	
			state = torch.Tensor(env.reset())

			# set camera trolly
			cam_trolly = camera_trolly()

			# set viewer for onscreen
			viewer_alive = True
			if is_render:
				env.sim.update_camera(cam_name='viewer')
				env.sim.viewer.update_hfield(0)
			else:
				env.sim.viewer_paused = False
			
			# set renderer for offscreen
			if is_record:
				env.sim.init_renderers()
				
			# episode counters
			done = False
			steps = 0
			returns = 0
			# set the mode latetnt 2 be tested

			while not done:
				
				if not env.sim.viewer_paused and viewer_alive:

								
					action = policy(state)
					next_state, reward, done, info_dict = env.step(action.numpy())
					# override the oracle if mode latent is given
					if mode_latent is not None:
						if env.this_epi_nstep >= env.exp_conf['oracle']['prediction_horizon']:
							env.curr_mode_latent = mode_latent
							# TMP: HARD CODED
							next_state[-2:] = torch.Tensor(mode_latent)

					# set custom camera pos for movable cameras
					cam_trolly.update()
					
					# onscreen rendering
					if is_render:
						time.sleep(0.02) # for mesh model
						env.sim.update_camera(cam_name='viewer' ,pos=cam_trolly.pos)
						env.sim.viewer.sync()

					# offscreen rendering
					if is_record:
						env.sim.update_camera(cam_name='free_camera' ,pos=cam_trolly.pos)

					state = torch.Tensor(next_state)
					steps += 1
					returns += reward

				if is_render:
					viewer_alive = env.sim.viewer.is_running()
					if not viewer_alive:
						break

			if not viewer_alive:
				break

			if is_record:				
				env.sim.delete_renderers()			
			np.savez(
					os.path.join(this_test_log_path,'metrics_log.npz'),
					epi_len = np.copy(steps),
					returns = np.copy(returns),
					mode_latent = np.copy(mode_latent),
					)
			env.close()

def eval_policy_tstg(args):

	tstng_conf_file = open(args.tstng_conf_path)
	tstng_conf = yaml.load(tstng_conf_file, Loader=yaml.FullLoader)
	tstng_conf_file.close()

	if exists_not_none('mode_parameter_test', tstng_conf['test_setup']):
		

		if isinstance(tstng_conf['test_setup']['mode_parameter_test']['paths2trngs'],list):
			paths2trngs = tstng_conf['test_setup']['mode_parameter_test']['paths2trngs']
		
		else:
			paths2trngs = [tstng_conf['test_setup']['mode_parameter_test']['paths2trngs']]

		nop = tstng_conf['test_setup']['mode_parameter_test']['nop']
		
		print("\nexps to be tested:")		
		for path2trngs in paths2trngs:
			print('\t',path2trngs)
		
		for path2trngs in paths2trngs:
			print("\nrunning mode param. test2 of:",path2trngs)
			# EXPERIMENT / POLICY DETAILS
			
			if exists_in('only_test', tstng_conf['test_setup']['mode_parameter_test']) and tstng_conf['test_setup']['mode_parameter_test']['only_test']:

				trngs = tstng_conf['test_setup']['mode_parameter_test']['only_test']
			else:
				trngs = os.listdir(path2trngs)
				trngs = [trng for trng in trngs if 'csv' not in trng]
			
			trngs = sorted(trngs, key=lambda e: int(e))


			# set mode parameter test env params 
			for ti,trng in enumerate(trngs):
				path2policy = path2trngs+trng+'/'
				print("\ntesting policy:",path2policy)

				trng_exp_conf_file = open(os.path.join(path2policy,'exp_conf.yaml')) # remove
				trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)

				# codebreak updates
				trng_exp_conf['env_entry'] = 'dtsd.envs.sim.biped_parkour.biped_env'

				# renders and record
				trng_exp_conf['sim_params']['render'] = False
				if exists_not_none('frame_recorder',tstng_conf):
					trng_exp_conf['frame_recorder'] = tstng_conf['frame_recorder']

				trng_exp_conf['visualize_reference']  = False
				# trng_exp_conf['sim_params']['model_path'] = 'dtsd/envs/rsc/models/mini_biped/xmls/biped_simple.xml'
				trng_exp_conf['sim_params']['model_path'] = 'dtsd/envs/rsc/models/mini_biped/xmls/biped_mvsc.xml'
				# common changes for all trials
				trng_exp_conf['terminations']['epi_steps_max_thresh'] = tstng_conf['test_setup']['mode_parameter_test']['max_epi_steps']
				trng_exp_conf['terminations']['total_reward_min_thresh'] = 0.0
				trng_exp_conf['terminations']['com_height_min_thresh'] = 0.3
				trng_exp_conf['terminations']['com_height_min_abs_thresh'] = 0.3
				trng_exp_conf['terminations'].pop('base_pos_x_ref_error_thresh')

				# trng_exp_conf['export_logger'] = {
				# 									'export_path': anlys_log_path,
				# 									'qpos': 'sim_data',
				# 									'qvel': 'sim_data',
				# 									'mode_latent': 'env',
				# 									'avoid_logfolder_names': True,
				# 									'export_conf': False,
				# 									}


				# trng_exp_conf['terminations']['base_pos_x_ref_error_thresh'] = 0.4

				
				# mediocre setting of reward for terminations
				trng_exp_conf['rewards'] = {
											'base_ori_error_exp_weight': 5,
											'base_pos_error_exp_weight': 5,
											'scales':
												{
												'base_ori_ref_error': 0.5,
												'base_pos_ref_error': 0.5,

												}

											}


				trng_exp_conf['task']['sample_type'] = tstng_conf['test_setup']['mode_parameter_test']['sample_type']
				trng_exp_conf['task']['track_x_length'] = tstng_conf['test_setup']['mode_parameter_test']['track_x_length']


				for tpk,tpv in tstng_conf['test_setup']['mode_parameter_test']['track_element'].items():
					print("\n\tcurrent mode tested:",tpk)
					trng_exp_conf['task']['track_elements'] = {

																
																tpk:{
																	'manipulate_terrain': tpv['manipulate_terrain'],
																	'param_names':tpv['param_names'],
																	'param_dist':
																	{
																		'type' : 'discrete',
																		'sampling': 'uniform',
																	},
																}
																}
					
					# print("running task parameter test of:",tpk)		
					param_len = len(tpv['param_names'])
					params_val_lists = []
					for i in range(param_len):
						params_val_lists.append(
													np.linspace(
																**tpv['param_'+str(i)],
													).tolist()
												)

							
					tstng_exp_confs = []
					for i,elem in enumerate(product(*params_val_lists)):

						elem = list(elem)							
						trng_exp_conf['task']['track_elements'][tpk]['param_dist']['points'] = [elem]
						tstng_exp_confs.append(copy.deepcopy(trng_exp_conf))
					
					n_trials_to_run = len(tstng_exp_confs)

					if nop > n_trials_to_run:
						nop = n_trials_to_run

					reminder = n_trials_to_run % nop
					quotient = (n_trials_to_run -reminder) / nop
					
					trial_index = 0
					processes = []
					tqdm.set_lock(multiprocessing.RLock())


					tpk = list(tstng_exp_confs[0]['task']['track_elements'].keys())[0]
					anlys_log_path = os.path.join(
						path2policy.replace('logs','results'),
						'mode_parameter_test_'+tstng_conf['test_setup']['mode_parameter_test']['test_name_suffix']+'/'+tpk+'/'
												)
					os.makedirs(anlys_log_path,exist_ok=True)


					for worker in range(nop):
							if reminder -worker > 0:
									n_trials_for_this_process = quotient + 1 
							
							else:
									n_trials_for_this_process = quotient    
							
							n_trials_for_this_process = int(n_trials_for_this_process)

							tstng_exp_confs_this_process = tstng_exp_confs[trial_index:trial_index+n_trials_for_this_process]				

							t = multiprocessing.Process(
														target=mode_parameter_test_worker, 
														args= (
																anlys_log_path,
																tstng_exp_confs_this_process, 
																path2policy,
																worker,
																trial_index
																
																),
														name= 't'+str(worker)
														)
							t.start()
							processes.append(t)
							trial_index += n_trials_for_this_process

					for t in processes:
						t.join()

	elif exists_not_none('n_rollouts_test', tstng_conf['test_setup']):

		paths2trngs = []
		if isinstance(tstng_conf['test_setup']['n_rollouts_test']['paths2trngs'],list):

			i = 0
			for path2trngs in tstng_conf['test_setup']['n_rollouts_test']['paths2trngs']:
				for ec in os.listdir(path2trngs):
					path2dir = os.path.join(path2trngs,direc)
					if os.path.isdir(path2dir):
						paths2trngs.append(path2dir)
						print(i,path2dir)
						i+=1
		
		else:
			paths2trngs = [tstng_conf['test_setup']['exp_log_path']]

		for path2trng in paths2trngs:
		

			n_trials_to_run = tstng_conf['test_setup']['n_rollouts_test']['n_rollouts']
			modelpath = os.path.join(path2trng,'actor.pt')	

			if os.path.isfile(modelpath):
				print('testing policy:',path2trng)

				anlys_log_path = modelpath.replace('logs','results').replace('actor.pt',
				'n_rollouts_test/'+str(n_trials_to_run)+'_rollouts_'+tstng_conf['test_setup']['n_rollouts_test']['test_name_suffix']+'/')		
				os.makedirs(anlys_log_path,exist_ok=True)
				

				nop = tstng_conf['test_setup']['n_rollouts_test']['nop']

				tstng_conf_name = args.tstng_conf_path.split('/')[-1]
				
				tstng_exp_conf_path = args.tstng_conf_path.replace(tstng_conf_name,'tstng_exp_conf.yaml')    

				if not os.path.isfile(tstng_exp_conf_path):
					tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
					tstng_exp_conf_file.close()

				trng_exp_conf_file = open(os.path.join(path2trng,'exp_conf.yaml')) # remove
				trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
				
				tstng_exp_conf_file = open(tstng_exp_conf_path)
				tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
				tstng_exp_conf_file.close()
				
				if args.trng_conf:
					tstng_exp_conf = trng_exp_conf
				
				else:
					tstng_exp_conf = trng_exp_conf
					tstng_exp_conf.update(tstng_conf)
					tstng_exp_conf.pop('test_setup')

				# overwrite render and record
				trng_exp_conf['sim_params']['render'] = tstng_conf['test_setup']['n_rollouts_test']['render']
				trng_exp_conf['sim_params']['record'] = tstng_conf['test_setup']['n_rollouts_test']['record']
				trng_exp_conf['visualize_reference']  = False
				
				if exists_not_none('frame_recorder',tstng_exp_conf):
					tstng_conf.pop('frame_recorder')
				
				if exists_not_none('export_logger',tstng_exp_conf):
					tstng_conf.pop('export_logger')

				# save conf in /exp_confs
				tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
				yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)
				
				# save con in /results	

				tstng_exp_conf_file =  open(os.path.join(anlys_log_path,'exp_conf.yaml'),'w')


				yaml.dump(
							tstng_exp_conf,
							tstng_exp_conf_file,
							default_flow_style=False,sort_keys=False)
				tstng_exp_conf_file.close()


				if nop > n_trials_to_run:
					nop = n_trials_to_run

				reminder = n_trials_to_run % nop
				quotient = (n_trials_to_run -reminder) / nop
				
				processes = []
				tqdm.set_lock(multiprocessing.RLock())

				for worker in range(nop):
						if reminder -worker > 0:
								n_trials_for_this_process = quotient + 1 
						else:
								n_trials_for_this_process = quotient    
						
						n_trials_for_this_process = int(n_trials_for_this_process)
						t = multiprocessing.Process(
													target=n_rollouts_test_worker, 
													args= (
															anlys_log_path,
															modelpath, 
															n_trials_for_this_process,
															tstng_exp_conf_path,
															worker,
															
															),
													name= 't'+str(worker)
													)
						t.start()
						processes.append(t)

				for t in processes:
					t.join()

			else:
				print('policy absent:',modelpath)

	elif exists_not_none('collect_rollouts',tstng_conf['test_setup']):
		paths2trngs = []
		if isinstance(tstng_conf['test_setup']['collect_rollouts']['paths2trngs'],list):

			i = 0
			for path2trngs in tstng_conf['test_setup']['collect_rollouts']['paths2trngs']:
				for direc in os.listdir(path2trngs):
					path2dir = os.path.join(path2trngs,direc)
					if os.path.isdir(path2dir):
						paths2trngs.append(path2dir)
						print(i,path2dir)
						i+=1
		
		else:
			paths2trngs = [tstng_conf['test_setup']['exp_log_path']]

		for path2trng in paths2trngs:
			n_trials_to_run = tstng_conf['test_setup']['collect_rollouts']['n_rollouts']
			modelpath = os.path.join(path2trng,'actor.pt')	

			if os.path.isfile(modelpath):
				print('testing policy:',path2trng)

				anlys_log_path = modelpath.replace('logs','results').replace('actor.pt',
				'collect_rollouts/'+str(n_trials_to_run)+'_rollouts_'+tstng_conf['test_setup']['collect_rollouts']['test_name_suffix']+'/')		
				os.makedirs(anlys_log_path,exist_ok=True)
				

				nop = tstng_conf['test_setup']['collect_rollouts']['nop']

				tstng_conf_name = args.tstng_conf_path.split('/')[-1]
				
				tstng_exp_conf_path = args.tstng_conf_path.replace(tstng_conf_name,'tstng_exp_conf.yaml')    

				if not os.path.isfile(tstng_exp_conf_path):
					tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
					tstng_exp_conf_file.close()

				trng_exp_conf_file = open(os.path.join(path2trng,'exp_conf.yaml')) # remove
				trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
				
				tstng_exp_conf_file = open(tstng_exp_conf_path)
				tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
				tstng_exp_conf_file.close()
				
				if args.trng_conf:
					tstng_exp_conf = trng_exp_conf
				
				else:
					tstng_exp_conf = trng_exp_conf
					tstng_exp_conf.update(tstng_conf)
					tstng_exp_conf.pop('test_setup')

				# overwrite render and record
				trng_exp_conf['sim_params']['render'] = tstng_conf['test_setup']['collect_rollouts']['render']
				trng_exp_conf['sim_params']['record'] = tstng_conf['test_setup']['collect_rollouts']['record']
				trng_exp_conf['visualize_reference']  = False
				
				if exists_not_none('frame_recorder',tstng_exp_conf):
					tstng_conf.pop('frame_recorder')
				
				if exists_not_none('export_logger',tstng_exp_conf):
					tstng_conf.pop('export_logger')

				# save conf in /exp_confs
				tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
				yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)
				
				# save con in /results	

				tstng_exp_conf_file =  open(os.path.join(anlys_log_path,'exp_conf.yaml'),'w')


				yaml.dump(
							tstng_exp_conf,
							tstng_exp_conf_file,
							default_flow_style=False,sort_keys=False)
				tstng_exp_conf_file.close()


				if nop > n_trials_to_run:
					nop = n_trials_to_run

				reminder = n_trials_to_run % nop
				quotient = (n_trials_to_run -reminder) / nop
				
				processes = []
				tqdm.set_lock(multiprocessing.RLock())

				for worker in range(nop):
						if reminder -worker > 0:
								n_trials_for_this_process = quotient + 1 
						else:
								n_trials_for_this_process = quotient    
						
						n_trials_for_this_process = int(n_trials_for_this_process)
						t = multiprocessing.Process(
													target=collect_rollouts_worker, 
													args= (
															anlys_log_path,
															modelpath, 
															n_trials_for_this_process,
															tstng_exp_conf_path,
															worker,
															
															),
													name= 't'+str(worker)
													)
						t.start()
						processes.append(t)

				for t in processes:
					t.join()

			else:
				print('policy absent:',modelpath)

	elif exists_not_none('lms_test_x0',tstng_conf['test_setup']):
		
		modelpath = os.path.join(tstng_conf['test_setup']['exp_log_path'],'actor.pt')	
		anlys_log_path = modelpath.replace('logs','results').replace('actor.pt','lms_test_x0_'+tstng_conf['test_setup']['lms_test_x0']['test_name_suffix']+'/logs/')
		os.makedirs(anlys_log_path,exist_ok=True)

		# load lme
		lme = load_lme_from_all_logs(tstng_conf['test_setup']['lms_test_x0']['n_rollout_logpath'])


		# save lme  and search space
		alp_till_logs = anlys_log_path.replace('logs/','')
		np.save(
				alp_till_logs+'lme',
				lme,
				)
		lme_center = np.mean(lme,axis=0)
		lme_r_max = np.max(np.linalg.norm(lme-lme_center,axis=1))
		print('lme_center:',lme_center)
		print('lme_r_max :',lme_r_max)
		r_ss = lme_r_max*tstng_conf['test_setup']['lms_test_x0']['search_space_scale']

		# search space square
		sss_xlim = [lme_center[0]-r_ss,lme_center[0]+r_ss]
		sss_ylim = [lme_center[1]-r_ss,lme_center[1]+r_ss]

		# span the search space
		latent_xs = np.linspace(
								sss_xlim[0],
								sss_xlim[1],
								tstng_conf['test_setup']['lms_test_x0']['latent_density_per_dim'],
								)						
		latent_ys = np.linspace(
								sss_ylim[0],
								sss_ylim[1],
								tstng_conf['test_setup']['lms_test_x0']['latent_density_per_dim'],
								)

		sss_points = [] # search space square points
		ssc_points = [] # search space circle points

		for x in latent_xs:
			for y in latent_ys:
				sss_points.append([x,y])
				if np.linalg.norm([x,y]-lme_center) <= r_ss:
					ssc_points.append([x,y])

		if tstng_conf['test_setup']['lms_test_x0']['search_space_type'] == 'square':
			all_mode_latents = sss_points
			ss_points = np.array(sss_points)
		else:
			all_mode_latents = ssc_points
			ss_points = np.array(ssc_points)

		# plot each search space in a subplot
		import matplotlib.pyplot as plt
		plt.scatter(lme[:,0],lme[:,1],s=1,marker='x', label='latent modes encountered')	
		plt.scatter(lme_center[0],lme_center[1],s=10, label='latent mode center')
		plt.scatter(ss_points[:,0],ss_points[:,1],s=1,marker='o', label='search space square')

		# annotate indices of ss_points next to them
		if tstng_conf['test_setup']['lms_test_x0']['latent_density_per_dim'] < 10:
			for i,point in enumerate(ss_points):
				plt.annotate(i,point)
		

		plt.grid()
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(alp_till_logs,'search_space.png'))
		plt.show()


		nop = tstng_conf['test_setup']['lms_test_x0']['nop']

		tstng_conf_name = args.tstng_conf_path.split('/')[-1]
		tstng_exp_conf_path = args.tstng_conf_path.replace(tstng_conf_name,'tstng_exp_conf.yaml')    

		if not os.path.isfile(tstng_exp_conf_path):
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
			tstng_exp_conf_file.close()

		trng_exp_conf_file = open(os.path.join(tstng_conf['test_setup']['exp_log_path'],'exp_conf.yaml')) # remove
		trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
		
		tstng_exp_conf_file = open(tstng_exp_conf_path)
		tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
		tstng_exp_conf_file.close()
		
		if args.trng_conf:
			tstng_exp_conf = trng_exp_conf
		
		else:
			tstng_exp_conf = trng_exp_conf
			tstng_exp_conf.update(tstng_conf)
			tstng_exp_conf.pop('test_setup')

		trng_exp_conf['sim_params']['render'] = False
		trng_exp_conf['visualize_reference']  = False
		# dont record for high density
		if tstng_conf['test_setup']['lms_test_x0']['latent_density_per_dim'] > 10:
			if exists_not_none('frame_recorder',tstng_exp_conf):
				tstng_conf.pop('frame_recorder')
		# code backward compatibility
		try:
			trng_exp_conf['terminations'] = { 
											'epi_steps_max_thresh': trng_exp_conf['oracle']['prediction_horizon']
											}
		except:
			trng_exp_conf['terminations'] = { 
											'epi_steps_max_thresh': trng_exp_conf['prev_rtm']['prediction_horizon']
											}
		if 'mode_latent' in trng_exp_conf['observations'].keys():
			trng_exp_conf['observations'].pop('mode_latent')
			trng_exp_conf['observations'].update({'custom_latent':None})

		trng_exp_conf['rewards'] = {
									'ctrl_mag_exp_weight': 0.01,
									'scales':
										{
										'ctrl_mag': 1.0,

										}
									}
		trng_exp_conf['export_logger'] = {
											'export_path': anlys_log_path,
											'qpos': 'sim_data',
											'qvel': 'sim_data',
											'mode_latent': 'env',
											'avoid_logfolder_names': True,
											'export_conf': False,
											}



		trng_exp_conf['task']['track_elements'] = {
													'flat':{
														'manipulate_terrain': False,
														'param_names':['goal_x'],
														'param_dist':
														{
															'type' : 'grid',
															'sampling': 'uniform',
															'support': [
																		[1.0],
																		[1.0],
																		],
														},
															}
													}
									


		# save conf in /exp_confs
		tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
		yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)
		

		n_trials_to_run = len(list(all_mode_latents))

		if nop > n_trials_to_run:
			nop = n_trials_to_run

		reminder = n_trials_to_run % nop
		quotient = (n_trials_to_run -reminder) / nop
		
		processes = []
		trial_index = 0
		tqdm.set_lock(multiprocessing.RLock())

		for worker in range(nop):
			if reminder -worker > 0:
					n_trials_for_this_process = quotient + 1 
			
			else:
					n_trials_for_this_process = quotient    
			
			n_trials_for_this_process = int(n_trials_for_this_process)

			mode_latents_this_process = all_mode_latents[trial_index:trial_index+n_trials_for_this_process]				
			t = multiprocessing.Process(
										target=lms_test_x0_worker, 
										args= (
												anlys_log_path,
												mode_latents_this_process,
												trng_exp_conf, 
												modelpath,
												worker,
												trial_index
												),
										name= 't'+str(worker)
										)
			t.start()
			processes.append(t)
			trial_index += n_trials_for_this_process
		
		for t in processes:
			t.join()

	elif exists_not_none('lms_test_tr',tstng_conf['test_setup']):
		
		modelpath = os.path.join(tstng_conf['test_setup']['exp_log_path'],'actor.pt')	
		anlys_log_path = modelpath.replace('logs','results').replace('actor.pt','lms_test_tr_'+tstng_conf['test_setup']['lms_test_tr']['test_name_suffix']+'/logs/')
		os.makedirs(anlys_log_path,exist_ok=True)
		# first do  single rollout test
		tstng_conf_name = args.tstng_conf_path.split('/')[-1]
		tstng_exp_conf_path = args.tstng_conf_path.replace(tstng_conf_name,'tstng_exp_conf.yaml')    

		if not os.path.isfile(tstng_exp_conf_path):
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
			tstng_exp_conf_file.close()

		trng_exp_conf_file = open(os.path.join(tstng_conf['test_setup']['exp_log_path'],'exp_conf.yaml')) # remove
		trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
		
		tstng_exp_conf_file = open(tstng_exp_conf_path)
		tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
		tstng_exp_conf_file.close()
		
		if args.trng_conf:
			tstng_exp_conf = trng_exp_conf
		
		else:
			tstng_exp_conf = trng_exp_conf
			tstng_exp_conf.update(tstng_conf)
			tstng_exp_conf.pop('test_setup')

		# for debugging
		trng_exp_conf['sim_params']['render'] = False
		trng_exp_conf['visualize_reference']  = False
		# dont record for high density
		if tstng_conf['test_setup']['lms_test_tr']['latent_density_per_dim'] > 10:
			if exists_not_none('frame_recorder',tstng_exp_conf):
				tstng_conf.pop('frame_recorder')
		# code backward compatibility
		try:
			trng_exp_conf['terminations'] = { 
											'epi_steps_max_thresh': int(2*trng_exp_conf['oracle']['prediction_horizon'])
											}
		except:
			trng_exp_conf['terminations'] = { 
											'epi_steps_max_thresh': int(2*trng_exp_conf['prev_rtm']['prediction_horizon'])
											}
		trng_exp_conf['rewards'] = {
									'ctrl_mag_exp_weight': 0.01,
									'scales':
										{
										'ctrl_mag': 1.0,

										}
									}
		trng_exp_conf['export_logger'] = {
											'export_path': anlys_log_path,
											'qpos': 'sim_data',
											'qvel': 'sim_data',
											'mode_latent': 'env',
											'avoid_logfolder_names': True,
											'export_conf': False,
											}
		trng_exp_conf['task'] = tstng_conf['test_setup']['lms_test_tr']['task']

		alp_till_logs = anlys_log_path.replace('logs/','')
		# save conf in /exp_confs
		tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
		yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)
		print('running rollout with oracle')
		processes = []

		t = multiprocessing.Process(
									target=lms_test_tr_worker, 
									args= (
											alp_till_logs,
											None,
											trng_exp_conf, 
											modelpath,
											0,
											"w_oracle"
											),
									name= 't_wo'
									)
		t.start()
		processes.append(t)
		
		

		
		# then do lms_test_x0 frm the transition set
		lme = load_lme_from_all_logs(tstng_conf['test_setup']['lms_test_tr']['n_rollout_logpath'])


		# save lme  and search space
		
		np.save(
				alp_till_logs+'lme',
				lme,
				)
		lme_center = np.mean(lme,axis=0)
		lme_r_max = np.max(np.linalg.norm(lme-lme_center,axis=1))
		print('lme_center:',lme_center)
		print('lme_r_max :',lme_r_max)
		r_ss = lme_r_max*tstng_conf['test_setup']['lms_test_tr']['search_space_scale']

		# search space square
		sss_xlim = [lme_center[0]-r_ss,lme_center[0]+r_ss]
		sss_ylim = [lme_center[1]-r_ss,lme_center[1]+r_ss]

		# span the search space
		latent_xs = np.linspace(
								sss_xlim[0],
								sss_xlim[1],
								tstng_conf['test_setup']['lms_test_tr']['latent_density_per_dim'],
								)						
		latent_ys = np.linspace(
								sss_ylim[0],
								sss_ylim[1],
								tstng_conf['test_setup']['lms_test_tr']['latent_density_per_dim'],
								)

		sss_points = [] # search space square points
		ssc_points = [] # search space circle points

		for x in latent_xs:
			for y in latent_ys:
				sss_points.append([x,y])
				if np.linalg.norm([x,y]-lme_center) <= r_ss:
					ssc_points.append([x,y])

		if tstng_conf['test_setup']['lms_test_tr']['search_space_type'] == 'square':
			all_mode_latents = sss_points
			ss_points = np.array(sss_points)
		else:
			all_mode_latents = ssc_points
			ss_points = np.array(ssc_points)

		# plot each search space in a subplot
		import matplotlib.pyplot as plt
		plt.scatter(lme[:,0],lme[:,1],s=1,marker='x', label='latent modes encountered')	
		plt.scatter(lme_center[0],lme_center[1],s=10, label='latent mode center')
		plt.scatter(ss_points[:,0],ss_points[:,1],s=1,marker='o', label='search space square')

		# annotate indices of ss_points next to them
		if tstng_conf['test_setup']['lms_test_tr']['latent_density_per_dim'] < 10:
			for i,point in enumerate(ss_points):
				plt.annotate(i,point)
		

		plt.grid()
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(alp_till_logs,'search_space.png'))
		plt.show()


		nop = tstng_conf['test_setup']['lms_test_tr']['nop']
		n_trials_to_run = len(list(all_mode_latents))

		if nop > n_trials_to_run:
			nop = n_trials_to_run

		reminder = n_trials_to_run % nop
		quotient = (n_trials_to_run -reminder) / nop
		
		trial_index = 0
		tqdm.set_lock(multiprocessing.RLock())

		for worker in range(nop):
			if reminder -worker > 0:
					n_trials_for_this_process = quotient + 1 
			
			else:
					n_trials_for_this_process = quotient    
			
			n_trials_for_this_process = int(n_trials_for_this_process)

			mode_latents_this_process = all_mode_latents[trial_index:trial_index+n_trials_for_this_process]				
			
			t = multiprocessing.Process(
										target=lms_test_tr_worker, 
										args= (
												anlys_log_path,
												mode_latents_this_process,
												trng_exp_conf, 
												modelpath,
												worker,
												trial_index
												),
										name= 't'+str(worker)
										)
			t.start()
			processes.append(t)
			trial_index += n_trials_for_this_process
		
		for t in processes:
			t.join()

	else:
		import matplotlib.pyplot as plt
		# default sigle test policy
		tstng_conf_name = args.tstng_conf_path.split('/')[-1]
		
		tstng_exp_conf_path = args.tstng_conf_path.replace(tstng_conf_name,'tstng_exp_conf.yaml')    
		
		if not os.path.isfile(tstng_exp_conf_path):
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
			tstng_exp_conf_file.close()

		trng_exp_conf_file = open(os.path.join(tstng_conf['test_setup']['exp_log_path'],'exp_conf.yaml')) # remove
		trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
		
		tstng_exp_conf_file = open(tstng_exp_conf_path)
		tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
		tstng_exp_conf_file.close()
		
		if args.trng_conf:
			tstng_exp_conf = trng_exp_conf
		
		else:
			tstng_exp_conf = trng_exp_conf
			tstng_exp_conf.update(tstng_conf)
			tstng_exp_conf.pop('test_setup')


		tstng_exp_conf['sim_params']['render'] = args.render_onscreen
		tstng_exp_conf['visualize_reference']  = args.visualize_reference
		

		if exists_not_none('frame_recorder',tstng_exp_conf):
			if '.mp4' not in tstng_exp_conf['frame_recorder']['export_path']:
				tstng_conf['frame_recorder']['export_path'] = os.path.join(
				tstng_conf['frame_recorder']['export_path'],
				tstng_conf['test_setup']['exp_log_path'].replace("./logs/","")
				)
			is_record = True
		else:
			is_record = False
		
		if exists_not_none('export_logger',tstng_exp_conf):
			tstng_conf['export_logger']['export_path'] = os.path.join(
			tstng_conf['export_logger']['export_path'],
			tstng_conf['test_setup']['exp_log_path'].replace("./logs/","")
			)

		tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
		yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)


		env = env_factory(tstng_exp_conf_path)()

		env.sim.set_default_camera()


		print("\ntesting policy:",tstng_conf['test_setup']['exp_log_path'],'\n')
		modelpath = os.path.join(tstng_conf['test_setup']['exp_log_path'],'actor.pt')
		policy = torch.load(modelpath)
		criticpath = os.path.join(tstng_conf['test_setup']['exp_log_path'],'critic.pt')
		critic = torch.load(criticpath)


		is_render = exists_and_true('render', env.sim.sim_params)



		with torch.no_grad():


			for n_epi in range(tstng_conf['test_setup']['n_episodes']):
				# reset	
				state = torch.Tensor(env.reset())

				# env.sim.viewer.sync()
				# time.sleep(50)

				# policy hidden state reset
				if hasattr(policy, 'init_hidden_state'):
					policy.init_hidden_state()
				if hasattr(critic, 'init_hidden_state'):
					critic.init_hidden_state()

				env.viewer_paused = True

				# set camera trolly
				cam_trolly = camera_trolly()

				# set viewer for onscreen
				viewer_alive = True
				

				env.sim.viewer.sync()
				if is_render:
					# env.sim.update_camera(cam_name='viewer')
					env.sim.viewer.update_hfield(0)
				else:

										
					env.sim.viewer_paused = False



				# set renderer for offscreen
				if is_record:
					env.sim.init_renderers()
					
				# episode counters
				done = False
				steps = 0
				returns = 0
				epi_st = time.time()


				qpos_traj = []
				qvel_traj = []
				toe_contact_state_traj = []
				state_predictions = []
				critic_values = []

				# toe contacts
				ltoe_contact = []
				rtoe_contact = []

				ltoe_contact_sched	= []
				rtoe_contact_sched	= []

				while not done:
					

					if not env.sim.viewer_paused and viewer_alive:

						# set custom camera pos for movable cameras
						base_pos = env.get_robot_base_pos()
						cam_trolly.update(subject_pos=base_pos)


						# offscreen rendering
						if is_record:
							env.sim.update_camera(
													cam_name='free_camera' ,
													pos=cam_trolly.pos,
													azim = cam_trolly.azim,
													elev = cam_trolly.elev,
													dist = cam_trolly.dist,							
													)						

						# onscreen rendering
						if is_render:
							time.sleep(env.dt)
							# time.sleep(0.1)
							# base_pos = env.get_robot_base_pos()
							# env.sim.update_camera(cam_name='viewer' ,pos=base_pos)
							# env.sim.update_camera(
							# 						cam_name='viewer' ,
							#  						pos=cam_trolly.pos,
							# 						azim = cam_trolly.azim,
							# 						elev = cam_trolly.elev,
							# 						dist = cam_trolly.dist,							
							# 					)
							# env.sim.viewer_paused = True
							env.sim.viewer.sync()


						# TODO: generalize as a feature for dynamics terrain later
						# if env.this_epi_nstep % int(env.exp_conf['oracle']['prediction_horizon']) == 0:
							
						# 	# get terrain height
						# 	env.get_toe_contact_state()
						# 	# if env.curr_toe_contact_state[0] or env.curr_toe_contact_state[1]:
						# 	base_pos = env.get_robot_base_pos()
						# 	terrain_height_bf = env.sim.get_terrain_height_at(base_pos)
						# 	robot_height_bf = base_pos[2] - terrain_height_bf
						# 	# print('terrain_height: bf',terrain_height)
						# 	env.sim.model.hfield_data[:] = 0
						# 	env.select_and_generate_track()
						# 	env.sim.viewer.update_hfield(0)
						# 	terrain_height = env.sim.get_terrain_height_at(base_pos)
						# 	env.sim.data.qpos[2] = terrain_height + robot_height_bf
						
						# 	# print('terrain_height: af',terrain_height, 'robot_height_bf:',robot_height_bf)
						
						action = policy(state)
						value = critic(state)
						next_state, reward, done, info_dict = env.step(action.numpy())

						
						# if env.this_epi_nstep < env.initial_balance_steps:
						# 	left_contact_schedule =  1
						# 	right_contact_schedule = 1
						# else:
						# 	time_now = (env.this_epi_nstep - env.initial_balance_steps) * env.dt
						# 	# a two phase gait
						# 	left_norm_t = time_now % env.exp_conf['rewards']['gait_period']
						# 	right_norm_t = (time_now + env.exp_conf['rewards']['phase_off_right_leg']) % env.exp_conf['rewards']['gait_period']

						# 	left_contact_schedule =  1 if left_norm_t < (env.exp_conf['rewards']['gait_period'] * env.exp_conf['rewards']['stance_percentage']) else 0
						# 	right_contact_schedule = 1 if right_norm_t < (env.exp_conf['rewards']['gait_period'] * env.exp_conf['rewards']['stance_percentage']) else 0

						# ltc, _ = env.sim.contact_bw_bodies(body1='terrain',body2='l_toe')
						# rtc, _ = env.sim.contact_bw_bodies(body1='terrain',body2='r_toe')

						# ltoe_contact.append(ltc)
						# rtoe_contact.append(rtc)
						# ltoe_contact_sched.append(left_contact_schedule)
						# rtoe_contact_sched.append(right_contact_schedule)



						qpos_traj.append(env.sim.data.qpos.copy())
						qvel_traj.append(env.sim.data.qvel.copy())
						critic_values.append(value.numpy())

						state = torch.Tensor(next_state)
						steps += 1
						returns += reward

					if is_render:
						viewer_alive = env.sim.viewer.is_running()
						if not viewer_alive:
							break

				if is_record:

					env.sim.delete_renderers()
				
				# tmp:
				# plt.plot(ltoe_contact, 'b',label='ltc_gtv')
				# plt.plot(ltoe_contact_sched,'b--',label='ltc_sched')

				# # plt.plot(rtoe_contact, 'r',label='rtc_gtv')
				# plt.plot(rtoe_contact_sched,'r--',label='rtc_sched')

				# plt.grid()
				# plt.legend()
				# plt.tight_layout()
				# plt.show()



				if not viewer_alive:
					break

				epi_et = time.time()
				

				if  exists_not_none('test_state_predictor',tstng_exp_conf):
					print(
							len(state_predictions),
							len(qpos_traj),
							len(qvel_traj),
							len(toe_contact_state_traj)
							)
					# make numpy arrays
					state_predictions = np.array(state_predictions)
					qpos_traj = np.array(qpos_traj)
					qvel_traj = np.array(qvel_traj)
					toe_contact_state_traj = np.array(toe_contact_state_traj)

					# plot state predictions vs true states
					base_pos_traj = qpos_traj[:,:3]
					base_quat_traj = qpos_traj[:,3:7]
					base_rpy_traj = np.zeros((len(base_quat_traj),3))
					for i,base_quat in enumerate(base_quat_traj):
						base_rpy_traj[i,:] = transformations.quat_to_euler(base_quat)
					base_tvel_traj = qvel_traj[:,:3]
					base_avel_traj = qvel_traj[:,3:6]
					
					state_true_traj = np.concatenate(
													(
														base_pos_traj,
														base_rpy_traj,
														base_tvel_traj,
														base_avel_traj,
														toe_contact_state_traj
													),
													axis=1
													)
					print("true traj shape:",state_true_traj.shape)
					print("pred traj shape:",state_predictions.shape)
					

					# round of state predictions 12, 13 
					state_predictions[:,12] = np.round(state_predictions[:,12])
					state_predictions[:,13] = np.round(state_predictions[:,13])

					# make an plt animation as a gif
					# plot state predictions vs true states
					make_time_series_animation(
						datas = [state_true_traj,state_predictions],
						titles = [
									'base_pos_0',	'base_pos_1',	'base_pos_2', 'base_rpy_0',	'base_rpy_1',	'base_rpy_2',
									'base_tvel_0',	'base_tvel_1',	'base_tvel_2',	'base_avel_0',	'base_avel_1',	'base_avel_2',
									'toe_contact_state_0',	'toe_contact_state_1',
									],
						save_path=env.frame_recorder.conf['export_path']+'/state_preds_vs_true.mp4',
						)
				print(
						'epi #',1+n_epi,
						'returns:',np.round(returns,2), 
						' of t=',round(steps*env.dt,3), 
						'mean:',np.round(returns/steps,2), 
						"compute time:",np.round(epi_et - epi_st,4)
					)


		env.close()

  # external logging while training # to remove soon

# placehoder for logging stuff
def log_stuff(self,iteration): 
	pass
	# # helper function to log internal env variables when called from outside
	# if exists_not_none('logger',self.exp_conf):
		
	# 	# saved only as numpy arrays for now
	# 	for data_name in self.exp_conf['logger']:
	# 	data = getattr(self,data_name)
	# 	np.savez_compressed(
	# 						self.exp_conf['logdir']+'/'+data_name+'/'+str(iteration)+'.npz',
	# 						data
	# 						)

def makedirs_for_log_stuff(self):
	pass