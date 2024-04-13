
'''
script to collect metrics over n_rolouts from a trained policy in its trianing domain (by default) 
or in a different domain if specified in the config file

TODO:
* use internal frame recorer to record videos ?

'''

import sys
sys.path.append('./')
import torch
from src.util import env_factory
from src.misc_funcs import load_key_from_all_logs
import os
import numpy as np
import yaml
import time
from tqdm.auto import tqdm, trange
import multiprocessing
from dtsd.envs.src.misc_funcs import *
import mediapy as media
import argparse

class camera_trolly:
	def __init__(self):

		self.pos = np.array([0.0,0,0.5])
		self.azim = 135 
		self.elev = -20 
		self.dist = 0.75 
		self.delta_pos = np.array([0.025,0.0,0.0])
		self.delta_dist = 0.105
		
	
	def update(self, subject_pos=None):

		if subject_pos is not None:
			self.pos[0] = subject_pos[0]
		else:
			self.pos += self.delta_pos
		pass


expid2label = {
					0: '[xt, zt, ct]',
					1: '[xt, zt]',
					2: '[xt, ht, ct]',
					3: '[xt, ht]',
					4: '[xt, ht, zt, ct]',
					5: '[xt, ht, zt]',
				}

def n_rollouts_test_worker(
					anlys_log_path,
					policy_path,
					n_rollouts,
					tstng_exp_conf_path,
					worker_id,
					):
		
		
		anlys_log_path += '/worker_'+str(worker_id)+'/'
		os.makedirs(anlys_log_path,exist_ok=True)
		
		np.random.seed(worker_id)
		env = env_factory(tstng_exp_conf_path)()
		is_render = exists_and_true('render', env.sim.sim_params)
		is_record = exists_and_true('record', env.sim.sim_params)
		policy = torch.load(policy_path)

		iteration_times = []


		# common
		episode_length = []
		undisc_returns = []

		# for agility
		base_tvel_maxs = []
		base_tacc_x_maxs = [] 

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
					
					if hasattr(policy, 'init_hidden_state'):
						policy.init_hidden_state()

					# set camera trolly
					cam_trolly = camera_trolly()

					# set viewer for onscreen
					viewer_alive = True
					if is_render:
						env.sim.update_camera(cam_name='viewer')
						env.sim.viewer.update_hfield(0)
						if env.sim.viewer_paused:
							print('viewer paused, press space to unpause')
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
					base_taccs = []

					while not done:
						if not env.sim.viewer_paused and viewer_alive:

							action = policy(state)
							next_state, reward, done, info_dict = env.step(action.numpy()) 
						
							# set custom camera pos for movable cameras
							cam_trolly.update()
							
							# onscreen rendering
							if is_render:
								time.sleep(env.dt) 
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
							base_tvel = env.get_robot_base_tvel().copy()
							base_tacc = env.get_robot_base_tacc().copy()
							# append data
							base_tvels.append(base_tvel)
							base_taccs.append(base_tacc)

							state = torch.Tensor(next_state)
							steps += 1
							returns += reward
						
						if is_render:
							viewer_alive = env.sim.viewer.is_running()
							if not viewer_alive:
								break

					base_tvels = np.array(base_tvels)
					base_taccs = np.array(base_taccs)
					# log performance metrics

					episode_length.append(steps)
					undisc_returns.append(returns)

					base_tvel_maxs.append(np.max(base_tvels[:,0]))
					base_tacc_x_maxs.append(np.max(np.linalg.norm(base_taccs,axis=1)))	

				
					iteration_times.append(time_elapsed - prev_time)
					prev_time = time_elapsed
					
					if not viewer_alive:
						break
					if is_record:
						media.write_video(
											os.path.join(anlys_log_path,'eps_'+str(n_epi)+'.mp4'),
											frames ,
											fps=int(1/env.dt),
											codec= 'hevc',
										)
						env.sim.delete_renderers()
				
		np.savez_compressed(
								os.path.join(anlys_log_path,'logs.npz'),
								
								# all metrics
								episode_length = np.copy(episode_length),
								undisc_returns = np.copy(undisc_returns),
								iteration_times = np.copy(iteration_times),
								mode_latents_encountered = np.copy(mode_latents_encountered),
								base_tvel_maxs = np.copy(base_tvel_maxs),
								base_tacc_x_maxs = np.copy(base_tacc_x_maxs),

							)
		env.close()


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser()
	parser.add_argument("--confpath", default='exp_confs/n_rollout_test.yaml', type=str)
	args = parser.parse_args()
	tstng_conf = yaml.load(open(args.confpath), Loader=yaml.FullLoader)

	
	for path2policy in tstng_conf['paths2policies']:
		
		modelpath = os.path.join(path2policy,'actor.pt')	
		if os.path.isfile(modelpath):
			print('****************************************************************************************************')

			expid = int(path2policy.split('/')[-2].split('_')[-1])

			print('testing policy:',path2policy,':',expid2label[expid])
			n_trials_to_run = tstng_conf['n_rollouts']
			anlys_log_path = modelpath.replace('logs','results').replace('actor.pt',
			'n_rollouts_test/'+str(n_trials_to_run)+'_rollouts_'+tstng_conf['test_name_suffix']+'/')		
			os.makedirs(anlys_log_path,exist_ok=True)
			

			nop = tstng_conf['nop']

			tstng_conf_name = args.confpath.split('/')[-1]
			tstng_exp_conf_path = args.confpath.replace(tstng_conf_name,'tstng_exp_conf.yaml')    

			if not os.path.isfile(tstng_exp_conf_path):
				tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
				tstng_exp_conf_file.close()

			# load training conf
			trng_exp_conf_file = open(os.path.join(path2policy,'exp_conf.yaml')) # remove
			trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)

			# male a placeholder for testing conf
			tstng_exp_conf_file = open(tstng_exp_conf_path)
			tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
			tstng_exp_conf_file.close()
			
			# update default trianing conf with any custom conf
			tstng_exp_conf = trng_exp_conf
			tstng_exp_conf.update(tstng_conf)

			# overwrite render and record condiguration
			trng_exp_conf['sim_params']['render'] = tstng_conf['render']
			trng_exp_conf['sim_params']['record'] = tstng_conf['record']
			trng_exp_conf['visualize_reference']  = False
			
			if exists_not_none('frame_recorder',tstng_exp_conf):
				tstng_conf.pop('frame_recorder')
			
			if exists_not_none('export_logger',tstng_exp_conf):
				tstng_conf.pop('export_logger')

			# save conf in /exp_confs
			tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
			yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)
			
			# save a conf copy in /results	
			tstng_exp_conf_file =  open(os.path.join(anlys_log_path,'exp_conf.yaml'),'w')
			yaml.dump(
						tstng_exp_conf,
						tstng_exp_conf_file,
						default_flow_style=False,sort_keys=False
					)
			tstng_exp_conf_file.close()

			# run rollouts across workers
			if tstng_conf['render']:
				print('since rendering is enabled, running a single worker')
				nop = 1
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

			base_tacc_x_maxs = load_key_from_all_logs(anlys_log_path,'base_tacc_x_maxs')
			base_tvel_maxs = load_key_from_all_logs(anlys_log_path,'base_tvel_maxs')
			episode_length = load_key_from_all_logs(anlys_log_path,'episode_length')
			max_fraudes = np.power(base_tvel_maxs,2) / (9.81*0.459) # 0.459 is the leg length, 9.81 is the gravity
			latetent_modes_encountered = load_key_from_all_logs(anlys_log_path,'mode_latents_encountered')
			
			
			if latetent_modes_encountered[0] is not None:
				lme_center = np.mean(latetent_modes_encountered,axis=0)
				lme_radius_max = np.max(np.linalg.norm(latetent_modes_encountered-lme_center,axis=1))
				print('\nM.T.A: {0:.2f}g, M.H.S: {1:.2f}, M.F: {2:.2f}, E.L: {3:.2f}, a_ID: [{4:.2f}, {5:.2f}], d_ID: {6:.2f}'.format(np.mean(base_tacc_x_maxs)/9.81,np.mean(base_tvel_maxs),np.mean(max_fraudes),np.mean(episode_length),lme_center[0],lme_center[1],lme_radius_max))
			else:
				print('\nM.T.A: {0:.2f}g, M.H.S: {1:.2f}, M.F: {2:.2f}, E.L: {3:.2f}, a_ID: NA, d_ID NA '.format(np.mean(base_tacc_x_maxs)/9.81,np.mean(base_tvel_maxs),np.mean(max_fraudes),np.mean(episode_length),np.mean(base_tacc_x_maxs),np.mean(base_tvel_maxs)))

		else:
			print('policy absent:',modelpath)
	


