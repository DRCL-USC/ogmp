import os,sys
sys.path.append('./')

import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from .ae_trainer import get_device
import importlib
from itertools import product
import random


custom_tight_layout = lambda  :     plt.subplots_adjust(
						left = 0.05,
						right= 0.95,
						bottom=0.05,
						top=0.94,
						hspace=0.1,
						wspace=0.155,
						)

PREV_RTM_X0_NOMINAL =   [
						  0.0, 0.0, 0.0,
						  0.0, 0.0, 0.5,
						  0.0, 0.0, 0.0,
						  0.0, 0.0, 0.0,
							-9.8
						]

PREV_RTM_X0_LABELS = [
					  'roll', 'pitch', 'yaw', 
					  'x', 'y', 'z',
					  'roll_dot', 'pitch_dot', 'yaw_dot',
					  'x_dot', 'y_dot', 'z_dot',
					  'g'
					  ] 

custom_tight_layout = lambda  :     plt.subplots_adjust(
						left = 0.05,
						right= 0.95,
						bottom=0.05,
						top=0.94,
						hspace=0.1,
						wspace=0.155,
						)
def downsample(data, ds_rate=10):
    return data[::ds_rate,:]

def generate_random_colors(n):
    # Generate 'n' random colors
    random_colors = []
    for _ in range(n):
        red = random.uniform(0.1, 1.0)
        green = random.uniform(0.1, 1.0)
        blue = random.uniform(0.1, 1.0)
        random_colors.append((red, green, blue))
    return random_colors

def load_key_from_all_logs(path2logs,key):
	loglist = os.listdir(path2logs)

	key_vals = None
	for i,log_name in enumerate(loglist):
		
		if 'worker_' in log_name and os.path.isdir(os.path.join(path2logs,log_name)):

			worker_log = np.load(os.path.join(path2logs,log_name+'/logs.npz'),allow_pickle=True)

			if key_vals is None:
				key_vals = worker_log[key]
			else:
				key_vals = np.concatenate((key_vals,worker_log[key]))
	return key_vals

def load_lme_from_all_logs(path2logs):
	loglist = os.listdir(path2logs)


	lme = None
	for i,log_name in enumerate(loglist):
		
		if 'worker_' in log_name and os.path.isdir(os.path.join(path2logs,log_name)):

			worker_log = np.load(os.path.join(path2logs,log_name+'/logs.npz'),allow_pickle=True)
			if lme is None:
				lme = worker_log['mode_latents_encountered']
			else:
				lme = np.vstack((lme,worker_log['mode_latents_encountered']))

	return lme


def make_time_series_animation(datas, titles=None, save_path='results/ts_animation.mp4'):
	n_steps = datas[0].shape[0]
	
	data_dim = datas[0].shape[1]
	n_rows = 2
	n_cols = datas[0].shape[1] // n_rows
	
	# if tiles are not provided, use default titles
	if titles is None:
		titles = [f'dim {i}' for i in range(data_dim)]
	# Create subplots
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5))
	axs = axs.flatten()
	for i,ax in enumerate(axs):
		ax.set_title(titles[i])
		ax.set_xlim(0, n_steps)
		# min and max of all datas along this dimension
		# ax.set_ylim(np.min([np.min(data[:,i]) for data in datas]), np.max([np.max(data[:,i]) for data in datas]))
		ax.set_ylim(-1.5, 1.5)
		ax.grid()

	fig.tight_layout()
	# Initialize lines for each subplot, # plot each data as a different line
	all_lines = []
	for data in datas:
		for dim in range(data.shape[1]):
			line, = axs[dim].plot([], [], lw=2)
			all_lines.append(line)

	# Function to update the plot at each time step
	def update(frame):
		for i,line in enumerate(all_lines):
			data_i = i// data_dim
			dim_i = i%data_dim
			line.set_data(range(frame), datas[data_i][:frame,dim_i])
		return all_lines
	
	# Create animation
	ani = FuncAnimation(fig, update, frames=n_steps, blit=True)
	# save the animation as an mp4. 
	ani.save(save_path)

def get_all_param_combs(mode_conf):
	
	param_len = len(mode_conf['param_names'])
	params_val_lists = []
	for i in range(param_len):
		params_val_lists.append(
									np.linspace(
												**mode_conf['param_'+str(i)],
									).tolist()
								)


	return list(product(*params_val_lists))

def load_trajs(
				conf,
				):

	traj_names = []
	train_set = []
	print('loaded trajs:')
	for traj_name in conf['trajs_used']:
		# traj_name = 'hop'
		if '.yaml' not in traj_name and '.png' not in traj_name:
			traj_datapath = conf['traj_folderpath']+traj_name+'.npz'
			
			data_file1 = np.load(traj_datapath)
			
			if conf['input_type'] == 'base_contact':

				merged_data = np.append(
										data_file1['qpos_traj'][:,0:7],
										data_file1['qvel_traj'][:,0:6],
										# data_file1['cont_seq'][:,:],                                    
										axis=1
										).astype('float32')

				merged_data = np.append(
										merged_data,
										data_file1['cont_seq'],
										axis=1
										).astype('float32')
			
			elif conf['input_type'] == 'base_only':
				merged_data = np.append(
										data_file1['qpos_traj'][:,0:7],
										data_file1['qvel_traj'][:,0:6],
										axis=1).astype('float32')

			elif conf['input_type'] == 'full_state' :
				merged_data = np.append(data_file1['qpos_traj'],data_file1['qvel_traj'],axis=1).astype('float32')
			else:
				print("requested input type absent")
				exit()
			# print("traj dim",merged_data.shape)
			train_set.append(torch.from_numpy(merged_data))
			traj_name = traj_name.replace(".npz","")
			print('\t',traj_name)
			traj_names.append(traj_name)
	
	return train_set, traj_names

def get_mode_params(key,x0,param,terrain_res):
	if key == 'flat':
		terrain_map = np.array(np.zeros((1,terrain_res)))
		state_goal = np.array([[
								0.0, 0.0, 0.0, 
								float(x0[3])+param[0], 0.0, 0.5, 
								0.0, 0.0, 0.0,
								0.0, 0.0, 0.0, 
								-9.8
								]]).T 
	elif key in ['gap','block']:
		
		t_start = param[0]
		t_end = param[0] + param[1]
		t_height = param[2]

		terrain_map = np.array(np.zeros((1,terrain_res)))
		terrain_map[0,int(t_start*terrain_res):int(t_end*terrain_res)] = t_height

		state_goal = np.array([[
								0.0, 0.0, 0.0, 
								float(x0[3])+1.0, 0.0, 0.5, 
								0.0, 0.0, 0.0,
								0.0, 0.0, 0.0, 
								-9.8
								]]).T 

	elif key == 'yaw_ip':
		terrain_map = np.array(np.zeros((1,terrain_res)))
		state_goal = np.array([[
								0.0, 0.0, float(x0[2]) + param[0], 
								0.0, 0.0, 0.5, 
								0.0, 0.0, 0.0,
								0.0, 0.0, 0.0, 
								-9.8
								]]).T 
	elif key == 'yaw_frwd':
		terrain_map = np.array(np.zeros((1,terrain_res)))
		state_goal = np.array([[
								0.0, 0.0, float(x0[2]) + param[1], 
								float(x0[3])+param[0], 0.0, 0.5, 
								0.0, 0.0, 0.0,
								0.0, 0.0, 0.0, 
								-9.8
								]]).T 
	
	elif 'flip' in key:
		h0 = param[0]
		x0[5] += h0
		state_goal = None
		terrain_map = None  

	else:
		print("requested input type absent")
		exit()
	
	return state_goal, terrain_map, x0

def load_trajs_frm_preview(
							conf,
							verbose=False,
							return_param_combs=False,
							device=None
							):

	if device is None:
		device = get_device()

	prev_class_name = conf['prev_rtm']['entry'].split('.')[-1]
	prev_file_entry = conf['prev_rtm']['entry'].replace('.'+prev_class_name,'')
	
	prev_module = importlib.import_module(prev_file_entry)

	prev_rtm = getattr(prev_module,prev_class_name) 

	traj_names = []
	train_set = []
	pctg = prev_rtm(conf['prev_rtm']['terrain_map_resolution'])

	one_by_terrain_res = int(1/conf['prev_rtm']['terrain_map_resolution'][0])
	print('loaded trajs:')
	
	for key in conf['tasks_for_trng'].keys():
		
		# state_init = np.array([PREV_RTM_X0_NOMINAL]).T

		x0s = []

		
		if 'vary_x0' not in conf['tasks_for_trng'][key].keys():
			x0s.append(PREV_RTM_X0_NOMINAL)
		else:
			x0s_options = []
			for dof, dof_xo_nom in zip(PREV_RTM_X0_LABELS,PREV_RTM_X0_NOMINAL):
				if dof in conf['tasks_for_trng'][key]['vary_x0']['dofs']:
					index_of_dof = conf['tasks_for_trng'][key]['vary_x0']['dofs'].index(dof)
					x0s_options.append(np.linspace(
													start=conf['tasks_for_trng'][key]['vary_x0']['val_lims'][index_of_dof][0],
													stop=conf['tasks_for_trng'][key]['vary_x0']['val_lims'][index_of_dof][1],
													num=conf['tasks_for_trng'][key]['vary_x0']['n_samples'][index_of_dof]
													).tolist()
										)
				else:
					x0s_options.append([dof_xo_nom])


			for element in product(*x0s_options):
				x0s.append(list(element))


		print(key,"total # of x0s:",len(x0s))


		mode_conf = conf['tasks_for_trng'][key]
		all_param_combs = get_all_param_combs(mode_conf)

		try:
			prediction_horizon = conf['prev_rtm']['prediction_horizon']
		except:
			prediction_horizon = None

		for x0_i,x0 in enumerate(x0s):
			# print('x0:',x0)
			for pi, param in enumerate(all_param_combs):

				state_init = np.array([x0]).T

				state_goal, terrain_map, state_init = get_mode_params(
																key,
																state_init,
																param,
																one_by_terrain_res
																)
				# print('#################')
				# print('\t',key,'param:',param)
				# print('\t\t',state_init.T)

				if 'flip' in key:
					x_sol,u_sol,preview_q_pos,preview_q_vel,t_traj = pctg.traj_opt(
																					state_init,
																					key,
																					list(param)
																				)                
				else:
					x_sol,u_sol,preview_q_pos,preview_q_vel,t_traj = pctg.traj_opt(
																					state_init,
																					state_goal,
																					terrain_map,
																					yorp_rot=key,
																				)
					
				if prediction_horizon is not None:
					merged_data = np.append(
											preview_q_pos[:prediction_horizon,0:7],
											preview_q_vel[:prediction_horizon,0:6],
											axis=1).astype('float32')
				else:                
					merged_data = np.append(
											preview_q_pos[:,0:7],
											preview_q_vel[:,0:6],
											axis=1).astype('float32')                   
			
				train_set.append(torch.from_numpy(merged_data).to(device=device))
				traj_name = key+'_v'+str(x0_i)+str(pi)
				print('\t',traj_name,merged_data.shape)
				traj_names.append(traj_name)
				if verbose:
					pctg.plotter(x_sol,u_sol)
					if 'flip' in key:
						pctg.animate(x_sol,u_sol,h_blk=param[0], fname=None)
					else:
						pctg.animate(x_sol,u_sol,fname=None)  

	if return_param_combs:
		all_param_combs = all_param_combs*len(x0s)
		return train_set, traj_names, all_param_combs
	else:
		return train_set, traj_names

def sample_frm_dataset(
						train_set,
						traj_names,
						n_samples= 4
						):


	train_set_sample = []
	traj_names_sample = []

	for i in range(n_samples):
		sample_i = np.random.randint(0,len(train_set))
		train_set_sample.append(train_set[sample_i])
		traj_names_sample.append(traj_names[sample_i])
	
	return train_set_sample, traj_names_sample

def plot_latents(
					zs,
					sample_names,
					savepath=None

				):
	plt.imshow(
				zs, 
				origin="upper", 
				interpolation="nearest"
				)

	plt.yticks(np.arange(-0.5, zs.shape[0]),[""]+sample_names)
	plt.xticks(np.arange(-0.5, zs.shape[1]),[""]*(zs.shape[1]+1),
					verticalalignment="bottom")

	plt.title('latents')
	plt.grid(color='black', linestyle='-', linewidth=1)
	plt.colorbar(orientation="vertical")

	# plt.tight_layout()
	fig = plt.gcf()
	fig.set_tight_layout(True)
	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath+'/latents.png', bbox_inches='tight',pad_inches = 0)
	# plt.show()
	plt.close()

def plot_latent_logits(
					zs_mean,
					zs_std,
					sample_names,
					savepath=None

				):
	
	fig, axs = plt.subplots(1,2)    
	axs[0].imshow(
					zs_mean, 
					origin="upper", 
					interpolation="nearest"
				)
	axs[0].set_title('mean')
	axs[1].imshow(
					zs_std, 
					origin="upper", 
					interpolation="nearest"
				)
	axs[1].set_title('std')
	for ax in axs:
		ax.set_yticks(np.arange(-0.5, zs_mean.shape[0]),[""]+sample_names)
		ax.set_xticks(np.arange(-0.5, zs_mean.shape[1]),[""]*(zs_mean.shape[1]+1),
					verticalalignment="bottom")

		ax.grid(color='black', linestyle='-', linewidth=1)
		fig.colorbar(ax.images[0], ax=ax, orientation='vertical')
		
	# plt.colorbar(orientation="vertical")
	fig.suptitle('latent logits')


	# plt.tight_layout()
	fig = plt.gcf()
	fig.set_tight_layout(True)
	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath+'/latents_logits.png', bbox_inches='tight',pad_inches = 0)
	# plt.show()
	plt.close()

def plot_2d_mode_space(
						zs_groups,
						groups_variant_names,
						annotate_variant_name=False,
						vary_variants_alpha=False,
						savepath=None,

						):
	


	for grp_i,grp_name in enumerate(groups_variant_names.keys()):
		zs = zs_groups[grp_name]['mean']
		for sample_i,((z_x,z_y), variant_name) in enumerate(zip(
																zs,
																groups_variant_names[grp_name]
															)):
			plt.scatter(
						z_x,z_y,
						color= 'C'+str(grp_i),
						label = grp_name if sample_i == 0 else None,
						alpha = 1 if not vary_variants_alpha else 1/(sample_i+1)
						)
			if annotate_variant_name:
				plt.text(z_x,z_y,variant_name,rotation=45) 

	
	plt.xlabel("latent_1")
	plt.ylabel("latent_2")

	# plt.xlim(-1,1)
	# plt.ylim(-1,1)

	plt.legend()
	plt.grid()
	plt.tight_layout()
	
	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath+'/mode_space.png')
	
	plt.close()

def plot_2d_mode_space_stats(
								zs_groups,
								groups_variant_names,
								annotate_variant_name=False,
								vary_variants_alpha=False,
								savepath=None,
								show_std=False,
								connect_grp_means=False,
								export_mode_space=False,

							):
	if export_mode_space:
		export_mode_space_dict = {}
	
	for grp_i,grp_name in enumerate(groups_variant_names.keys()):


		if connect_grp_means:
			zs = np.array(zs_groups[grp_name]['mean'])
			plt.plot(
						zs[:,0],
						zs[:,1],
						color= 'C'+str(grp_i),
						# label = grp_name,
						alpha = 1 
						)

		# just the mean
		if export_mode_space:
			export_mode_space_dict[grp_name] = np.array(zs_groups[grp_name]['mean'])
		
		for sample_i,((z_mean_x,z_mean_y),(z_std_x,z_std_y) ,variant_name) in enumerate(zip(
																							zs_groups[grp_name]['mean'],
																							zs_groups[grp_name]['std'],
																							groups_variant_names[grp_name]
																							)
																						):
			plt.scatter(
						z_mean_x,z_mean_y,
						color= 'C'+str(grp_i),
						label = grp_name if sample_i == 0 else None,
						alpha = 1 if not vary_variants_alpha else 1/(sample_i+1)
						)

			if show_std:

				plt.errorbar(
								z_mean_x,z_mean_y,
								xerr=z_std_x,
								yerr=z_std_y,
								color= 'C'+str(grp_i),
								alpha = 1 if not vary_variants_alpha else 1/(sample_i+1)
							)
			if annotate_variant_name:
				plt.text(z_mean_x,z_mean_y,variant_name,rotation=45) 

	
	plt.xlabel("latent_1")
	plt.ylabel("latent_2")

	xmin, xmax, ymin, ymax = plt.axis()
	# plt.xlim(-1,1)
	# plt.ylim(-1,1)

	plt.legend()
	plt.grid()
	plt.tight_layout()
	
	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath+'/mode_space.png')
	
	if export_mode_space:
		np.savez(
					savepath+'/mode_space.npz',
					**export_mode_space_dict,
					xlim = [xmin,xmax],
					ylim = [ymin,ymax],
				)




	plt.close()

def get_vae_reconstruction(
							model,
							sample,
						):
	if sample.dim() == 2:
		sample = sample.unsqueeze(0)
	_,sample_hat,z_stats,_ = model(sample)
	return sample_hat,z_stats

def get_ae_reconstruction(
							model,
							sample,
						):
	z_stats = get_encodings(model,[sample])

	sample_hat = model(sample.unsqueeze(0))
	return sample_hat,z_stats
	
def get_and_plot_reconstructions(
							model,
							samples,
							sample_names,
							savepath=None,
							is_vae=False,

):

	savepath = savepath+'/reconstructions/'
	os.makedirs(savepath,exist_ok=True)
	img_titels = ['x','x_hat']

	model.eval()
	for sample, sample_name in zip(samples,sample_names):
		
		if is_vae:
			sample_hat,_ = get_vae_reconstruction(model,sample)
		else:
			sample_hat,_ = get_ae_reconstruction(model,sample)

		sample_hat = sample_hat.squeeze(0)
		sample = sample.squeeze(0)

		sample = sample.detach().cpu().numpy()
		sample_hat = sample_hat.detach().cpu().numpy()

		imgs_to_plot = [sample,sample_hat]
		
		fig, axs = plt.subplots(len(imgs_to_plot),1)
		fig.suptitle(sample_name)




		v_min = min([img.min() for img in imgs_to_plot])
		v_max = max([img.max() for img in imgs_to_plot])

		img_plts = []
		for i, img2p in enumerate(imgs_to_plot):

			img2p = img2p.T
			img_plts.append( 
							axs[i].imshow(
											img2p, 
											interpolation="nearest",
											vmin=v_min,
											vmax=v_max
										)
							)
			axs[i].set_title(img_titels[i])
			axs[i].set_xticks(np.arange(-0.5, img2p.shape[1]))
			axs[i].set_yticks(np.arange(-0.5, img2p.shape[0]))

			axs[i].set_yticklabels( [""]*(img2p.shape[0]+1),
								verticalalignment="bottom")
			axs[i].set_xticklabels([""]*(img2p.shape[1]+1) )

			axs[i].grid(color='black', linestyle='-', linewidth=1)
		
		fig.colorbar(img_plts[-1], ax=axs[-1], orientation='horizontal')
		
		fig.tight_layout()

		if savepath is None:
			plt.show()
		else:

			plt.savefig(savepath+sample_name, bbox_inches='tight',pad_inches = 0)
		plt.close()

def get_vae_encodings(model, samples):
	
	zs = []
	z_means = []
	z_stds = []


	model.eval()
	for x in samples:
		x = x.unsqueeze(0)



		_,x_hat,z_stats,info = model(x)

		z = z_stats['sample']
		zs.append(z[0,0,:].detach().cpu().numpy())
		z_means.append(z_stats['mean'].detach().cpu().numpy()[0])
		z_stds.append(z_stats['std'].detach().cpu().numpy()[0])
	
	zs = np.array(zs)
	z_means = np.array(z_means)
	z_stds = np.array(z_stds)
	return {'sample':zs,'mean':z_means,'std':z_stds}

def get_encodings(model, samples):

	zs = []
	z_means = []
	z_stds = []

	model.eval()
	for x in samples:
		z = model.encoder(x)

		zs.append(z.detach().cpu().numpy())
		z_means.append(z.detach().cpu().numpy())
		z_stds.append(np.zeros(2))

	zs = np.array(zs)
	z_means = np.array(z_means)
	z_stds = np.array(z_stds)

	return {'sample':zs,'mean':z_means,'std':z_stds}


