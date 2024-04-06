'''
script to collect rollouts from a trained policy in its trianing domain (by default) 
or in a different domain if specified in the config file


TODO:
* use internal logger to log the data
* use internal frame recorer to record videos ?

'''

import sys
sys.path.append('./')
from dtsd.envs.src.misc_funcs import *
from tqdm.auto import tqdm, trange
from util import env_factory
import multiprocessing
import mediapy as media
import numpy as np
import argparse
import torch
import yaml
import time
import os




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


def collect_rollouts_worker(
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
				
				
				for n_epi in pbar:
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
					obs_traj = []
					act_traj= []
					qpos_traj = []
					qvel_traj = []
					while not done:
						if not env.sim.viewer_paused and viewer_alive:

							
							obs_traj.append(state.numpy())
							action = policy(state)
							act_traj.append(action.numpy())
							next_state, reward, done, info_dict = env.step(action.numpy()) 
						
							# next state
							qpos = env.sim.data.qpos.copy()
							qvel = env.sim.data.qvel.copy()
							
							
							qpos_traj.append(qpos)
							qvel_traj.append(qvel)

							# set custom camera pos for movable cameras
							cam_trolly.update()
							
							# onscreen rendering
							if is_render:
								time.sleep(env.dt) # for mesh model
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
								observations = np.copy(observations),
								actions = np.copy(actions),
								qposs = np.copy(qposs),
								qvels = np.copy(qvels),
								toe_contact_states = np.copy(toe_contact_states),
							)
		env.close()


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--confpath", default='exp_confs/collect_rollouts.yaml', type=str)
	args = parser.parse_args()
	tstng_conf = yaml.load(open(args.confpath), Loader=yaml.FullLoader)


	for path2policy in tstng_conf['paths2policies']:
		
		modelpath = os.path.join(path2policy,'actor.pt')	
		if os.path.isfile(modelpath):
			print('testing policy:',path2policy)
			n_trials_to_run = tstng_conf['n_rollouts']
			anlys_log_path = modelpath.replace('logs','results').replace('actor.pt',
			'collect_rollouts/'+str(n_trials_to_run)+'_rollouts_'+tstng_conf['test_name_suffix']+'/')		
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
