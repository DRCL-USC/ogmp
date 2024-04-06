import sys
sys.path.append("./")
import numpy as np
import importlib
import yaml
import time


# exp_conf_path = './dtsd/confs/parkour_env_test.yaml'
exp_conf_path = './exp_confs/recover_trng_li.yaml'

conf_file = open(exp_conf_path)
exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)

env_class_name = exp_conf['env_entry'].split('.')[-1]
env_file_entry = '.'.join(exp_conf['env_entry'].split('.')[:-1])
env_module = importlib.import_module(env_file_entry)
env_class = getattr(env_module,env_class_name)
env = env_class(exp_conf_path)
if env.sim.sim_params['render']:
    env.sim.viewer._render_every_frame = False
    env.sim.viewer._paused = True
    env.sim.viewer.cam.distance = 3
    cam_pos = [env.sim.data.qpos[0], 0.0, 0.75]
    for i in range(3):        
        env.sim.viewer.cam.lookat[i]= cam_pos[i] 
    env.sim.viewer.cam.elevation = -15
    env.sim.viewer.cam.azimuth = 180

print('qpos0:',env.sim.model.qpos0.round(2))
print('dim(obs):',env.observation_space.shape)
print('dim(act):',env.action_space.shape)

for n_epi in range(5):
    env.reset()
    if env.sim.sim_params['render']:
        env.sim.viewer.update_hfield(0)
    else:
        env.sim.viewer_paused = False
    step = 0    
    returns = 0
    if env.sim.sim_params['render']:
        env.sim.viewer.sync()
        print('qpos0:',env.sim.data.qpos.round(2))
        print('qvel0:',env.sim.data.qvel.round(2))
        time.sleep(5)
    while True:        
        action =  np.random.uniform(-1,1,env.action_space.shape)
        obs,rew,done,info = env.step(action)
        print(np.isnan(obs),np.isnan(rew))
        if env.sim.sim_params['render']:
            env.sim.viewer.sync()
        if done:
            break
        step+=1
        returns+=rew
    print(
            'epi #',1+n_epi,
            'returns:',np.round(returns,2), 
            ' of t=',round(step*env.dt,3), 
        )