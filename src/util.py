import torch
import hashlib
import os
import numpy as np
from collections import OrderedDict
import yaml

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