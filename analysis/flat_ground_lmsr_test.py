import sys
sys.path.append('./')
import torch
import os
import numpy as np
import yaml
import time
from tqdm.auto import tqdm, trange
import multiprocessing
from dtsd.envs.src.misc_funcs import *
from src.misc_funcs import *
from src.util import env_factory
import argparse



class camera_trolly:
	def __init__(self):

		self.pos = np.array([0.25,0,0.5])
		self.azim = 135 
		self.elev = -20 
		self.dist = 0.75 
		self.delta_pos = np.array([0.0,0.0,0.0])
		self.delta_dist = 0.105
		
	
	def update(self, subject_pos=None):

		if subject_pos is not None:
			self.pos[0] = subject_pos[0]
		else:
			self.pos += self.delta_pos
		pass

def flat_ground_lmsr_test_worker(
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
			if hasattr(policy, 'init_hidden_state'):
				policy.init_hidden_state()

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
						time.sleep(env.dt) 
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

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--confpath", default='exp_confs/flat_ground_lmsr_test.yaml', type=str)
	args = parser.parse_args()
	tstng_conf = yaml.load(open(args.confpath), Loader=yaml.FullLoader)


	modelpath = os.path.join(tstng_conf['exp_log_path'],'actor.pt')	
	anlys_log_path = modelpath.replace('logs','results').replace('actor.pt','flat_ground_lmsr_test_'+tstng_conf['test_name_suffix']+'/logs/')
	os.makedirs(anlys_log_path,exist_ok=True)

	# load lme
	lme = load_lme_from_all_logs(tstng_conf['n_rollout_logpath'])


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
	r_ss = lme_r_max*tstng_conf['search_space_scale']

	# search space square
	sss_xlim = [lme_center[0]-r_ss,lme_center[0]+r_ss]
	sss_ylim = [lme_center[1]-r_ss,lme_center[1]+r_ss]

	# span the search space
	latent_xs = np.linspace(
							sss_xlim[0],
							sss_xlim[1],
							tstng_conf['latent_density_per_dim'],
							)						
	latent_ys = np.linspace(
							sss_ylim[0],
							sss_ylim[1],
							tstng_conf['latent_density_per_dim'],
							)

	sss_points = [] # search space square points
	ssc_points = [] # search space circle points

	for x in latent_xs:
		for y in latent_ys:
			sss_points.append([x,y])
			if np.linalg.norm([x,y]-lme_center) <= r_ss:
				ssc_points.append([x,y])

	all_mode_latents = sss_points
	ss_points = np.array(sss_points)


	# plot each search space in a subplot
	import matplotlib.pyplot as plt


	plt.scatter(lme[:,0],lme[:,1],s=20,marker='x', label='latent modes encountered (in domain)', color='blue')
	plt.scatter(lme_center[0],lme_center[1],s=20, label='latent mode center', color='blue')
	plt.plot([lme_center[0],lme_center[0]+lme_r_max],[lme_center[1],lme_center[1]],color='blue',linestyle='--', label='in domain radius')
	ax = plt.gca()
	circle = plt.Circle(lme_center, lme_r_max, color='blue', fill=False, linestyle='--')
	ax.add_artist(circle)
	plt.scatter(ss_points[:,0],ss_points[:,1],s=1,marker='o', label='search space ', color='gray')
	# plot the ss radius vertival line
	plt.plot([lme_center[0],lme_center[0]],[lme_center[1],lme_center[1]-r_ss],color='gray',linestyle='--', label='search space radius')


	# annotate indices of ss_points next to them
	if tstng_conf['latent_density_per_dim'] < 10:
		for i,point in enumerate(ss_points):
			plt.annotate(i,point)
	
	plt.title('defined latent search space')
	plt.grid()
	plt.legend(loc='upper right')
	plt.xlabel('latent 0')
	plt.ylabel('latent 1')
	plt.xlim(sss_xlim)
	plt.ylim(sss_ylim)
	plt.tight_layout()
	
	plt.savefig(os.path.join(alp_till_logs,'search_space.png'))
	print('close figure to continue')
	plt.show()
	plt.close()




	tstng_conf_name = args.confpath.split('/')[-1]
	tstng_exp_conf_path = args.confpath.replace(tstng_conf_name,'tstng_exp_conf.yaml')    

	if not os.path.isfile(tstng_exp_conf_path):
		tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
		tstng_exp_conf_file.close()

	trng_exp_conf_file = open(os.path.join(tstng_conf['exp_log_path'],'exp_conf.yaml')) # remove
	trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
	
	tstng_exp_conf_file = open(tstng_exp_conf_path)
	tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
	tstng_exp_conf_file.close()
	
	# update the conf
	tstng_exp_conf = trng_exp_conf
	tstng_exp_conf.update(tstng_conf)

	trng_exp_conf['sim_params']['render'] = tstng_conf['render']
	trng_exp_conf['visualize_reference']  = False

	if tstng_conf['render']:
		print('since render is True, the test will be run in a single process')
		nop = 1
	else:
		nop = tstng_conf['nop']

	# dont record for high density
	if tstng_conf['latent_density_per_dim'] > 10:
		if exists_not_none('frame_recorder',tstng_exp_conf):
			tstng_conf.pop('frame_recorder')
	
	# make epside length equal to prediction horizon
	trng_exp_conf['terminations']['epi_steps_max_thresh'] = trng_exp_conf['oracle']['prediction_horizon']
									

	# to log the simulation states
	trng_exp_conf['export_logger'] = {
										'export_path': anlys_log_path,
										'qpos': 'sim_data',
										'qvel': 'sim_data',
										'mode_latent': 'env',
										'avoid_logfolder_names': True,
										'export_conf': False,
										}

	# save conf in /exp_confs
	tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
	yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)
	
	# distribute the trials across workers
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
									target=flat_ground_lmsr_test_worker,
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

	
	# collect all results
	logs =[obj for obj in  os.listdir(anlys_log_path) if os.path.isdir(os.path.join(anlys_log_path,obj))]
	logs = sorted(logs,key=lambda x: int(x.split('.')[0]))
	fig, axs = plt.subplots(1,4,figsize=(20,4))

	robot_trajs = []
	for i in tqdm(range(len(logs)),desc='collecting logs'):
		path2log = os.path.join(anlys_log_path,logs[i],'log.npz')
		log = np.load(path2log)
		robot_qpos = log['qpos'][:,QPOS2USE]
		robot_qvel = log['qvel'][:,QVEL2USE]
		robot_state = np.hstack((robot_qpos,robot_qvel))
		robot_state = downsample(robot_state,ds_rate=int(ENV_DT/DATA_DT))
		robot_trajs.append(robot_state)

		path2mlog = os.path.join(anlys_log_path,logs[i],'metrics_log.npz')
		mlog = np.load(path2mlog)

	robot_trajs = np.array(robot_trajs)
	# print('robot_trajs.shape:',robot_trajs.shape)
	# print('ss_points.shape:',ss_points.shape)


	# compute the Z(x), plot 0 and 2
	in_z_space = []
	robot_trajs_in_z_space = []
	lm_in_z_space = []
	for i,point in enumerate(ss_points):
		if robot_trajs[i,-1,2] <= 0.4:
			in_z_space.append(0)
			axs[0].scatter(point[0],point[1],s=20,marker='o',color='gray')
			axs[2].scatter(point[0],point[1],s=20,marker='o',color='gray')
		else:
			in_z_space.append(1)
			robot_trajs_in_z_space.append(robot_trajs[i])
			lm_in_z_space.append(point)
			axs[0].scatter(point[0],point[1],s=20,marker='o',color='red')


	robot_trajs_in_z_space = np.array(robot_trajs_in_z_space)
	lm_in_z_space = np.array(lm_in_z_space)

	# print('robot_trajs_in_z_space.shape:',robot_trajs_in_z_space.shape)
	# print('lm_in_z_space.shape:',lm_in_z_space.shape)

	# make a trjectory(2d) in to a vector(1d)
	data2cluster = robot_trajs_in_z_space.reshape(robot_trajs_in_z_space.shape[0],-1)
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_score
	
	n_cluster_range = np.arange(2,min(15,len(data2cluster)))
	elbow_scores = []
	silhoutte_scores = []

	for n_clusters in tqdm(n_cluster_range,desc='clustering'):
		kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init="auto").fit(data2cluster)
		elbow_scores.append(kmeans.inertia_)
		silhoutte_scores.append(silhouette_score(data2cluster, kmeans.labels_))
		# print('n_clusters:',n_clusters,'silhouette_score:',silhouette_score(data2cluster, kmeans.labels_))	
	
	optimal_n_clusters = n_cluster_range[np.argmax(silhoutte_scores)]
	print('optimal_n_clusters:',optimal_n_clusters)
	# plot 1: elbow and silhouette scores
	axs[1].plot(n_cluster_range,elbow_scores,label='elbow score')
	axs[1].set_xlabel('n_clusters')
	axs[1].set_ylabel('elbow')
	axs1_twin = axs[1].twinx()
	axs1_twin.plot(n_cluster_range,silhoutte_scores,label='silhouette score',color='orange')
	axs1_twin.set_ylabel('silhouette')
	# draw a vline at optimal_n_clusters
	axs[1].axvline(x=optimal_n_clusters, color='red', linestyle='--')



	kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42,n_init="auto").fit(data2cluster)
	mode_clusters = [[] for _ in range(optimal_n_clusters)]

	for i,label in enumerate(kmeans.labels_): 
		mode_clusters[label].append(i)
	
	# sort the clusters by length
	mode_clusters.sort(key=len)

	
	# generate unique colors for each cluster
	colors = generate_random_colors(optimal_n_clusters)


	for mc,col in zip(mode_clusters,colors):
		# points in the cluster
		points = lm_in_z_space[mc]

		axs[2].scatter(points[:,0],points[:,1],s=20,marker='o',color=col)

		robot_trajs_in_cluster = robot_trajs_in_z_space[mc]

		# plot the cluster trajs
		for rt in robot_trajs_in_cluster:
			axs[3].plot(rt[:,2],color=col, alpha=0.5)
	
	axs[3].set_xlabel('timesteps')
	axs[3].set_ylabel('base_height')
	titles = ['Z(x)' , 'elbow and eilhouette scores', 'clustered Z(x)', 'clustered base_height traj.']
	latent_plots = [0,2]
	
	for ax,title in zip(axs,titles):
		ax.set_title(title)
		if 'Z(x)' in title:
			ax.set_xlabel('latent 0')
			ax.set_ylabel('latent 1')
		else:
			ax.grid()
	
	fig.suptitle('flat_ground_lmsr_test')
	fig.tight_layout()
	fig.savefig(os.path.join(alp_till_logs,'flat_ground_lmsr_test.png'))
	print('close figure to exit')
	plt.show()

 
		