'''
a script to rollout, view and record a trained policy
'''
import sys
sys.path.append('./')
from dtsd.envs.src.misc_funcs import *
from src.util import env_factory
import argparse
import torch
import yaml
import time
import os

class camera_trolly:
	def __init__(self):
		self.pos =[1.0,0,0.5]
		self.azim = 90
		self.elev = -10 
		self.dist = 4.0
	
	def update(self, subject_pos=None):
		pass

def rollout_policy(
                    env, 
                    policy, 
                    sync_camera=False,
                    ):
    """
    rollout policy in environment 
    """
    # reset	
    obs = torch.Tensor(env.reset())
    
    # policy reset
    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()
    
    env.viewer_paused = True # pause the viewer

    # set camera trolly
    cam_trolly = camera_trolly()

    # set viewer for onscreen
    terminate_and_exit = False
    
    is_render =  exists_and_true('render',env.exp_conf['sim_params'])
    is_record = exists_not_none('frame_recorder',env.exp_conf)

    if is_render:
        # update the hfield
        env.sim.viewer.update_hfield(0)
    else:
        # start running the simulation directly
        env.sim.viewer_paused = False
   
    # set offscreen renderer for recording
    if is_record:
        env.sim.init_renderers()

    # episode counters
    done = False
    steps = 0
    returns = 0
    
    while not done:
        # simulate if not paused
        if not env.sim.viewer_paused:
            # set cameras and sync with viewer
            base_pos = env.get_robot_base_pos()
            cam_trolly.update(subject_pos=base_pos)
            if is_record and sync_camera:
                env.sim.update_camera(
                                        cam_name='free_camera' ,
                                        pos=cam_trolly.pos,
                                        azim = cam_trolly.azim,
                                        elev = cam_trolly.elev,
                                        dist = cam_trolly.dist,							
                                   )	
            if is_render:
                time.sleep(env.dt)
                if sync_camera:
                    env.sim.update_camera(
                                            cam_name='viewer' ,
                                            pos=cam_trolly.pos,
                                            azim = cam_trolly.azim,
                                            elev = cam_trolly.elev,
                                            dist = cam_trolly.dist,							
                                        )
                env.sim.viewer.sync()
            
            action = policy(obs)
            next_obs, reward, done, info_dict = env.step(action.numpy())
            obs = torch.Tensor(next_obs)
            steps += 1
            returns += reward

        # terminate if viewer is closed
        if is_render and not env.sim.viewer.is_running():
            terminate_and_exit = True
            break

    # delete renderers to prevent glitch
    if is_record:
        env.sim.delete_renderers()

    print(
            'epi #:',f'{1+n_epi:03}',
            'returns:',f'{returns:0.2e}', 
            'of steps:',f'{steps:04}',  
            'in time:',f'{steps*env.dt:0.2f}',
        )    
    return terminate_and_exit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tstng_conf_path",  default="./exp_confs/parkour_test.yaml", type=str)  # path to testing log with policy and exp conf file
    parser.add_argument("--render_onscreen",          action='store_true')
    parser.add_argument("--visualize_reference",          action='store_true')
    parser.add_argument("--visualize_oracle",          action='store_true')
    parser.add_argument("--run_trng_conf",          action='store_true')
    parser.add_argument("--sync_camera",          action='store_true')
    args = parser.parse_args()

    # load the test setup
    tstng_conf_file = open(args.tstng_conf_path)
    tstng_conf = yaml.load(tstng_conf_file, Loader=yaml.FullLoader)
    tstng_conf_file.close()

    # make a temporary testing experiment config file
    tstng_conf_name = args.tstng_conf_path.split('/')[-1]
    tstng_exp_conf_path = args.tstng_conf_path.replace(tstng_conf_name,'tstng_exp_conf.yaml')
    if not os.path.isfile(tstng_exp_conf_path):
        tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
        tstng_exp_conf_file.close()
    
    # load the testing experiment config file
    tstng_exp_conf_file = open(tstng_exp_conf_path)
    tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)    
    tstng_exp_conf_file.close()

    # load the training experiment config file
    trng_exp_conf_file = open(os.path.join(tstng_conf['exp_log_path'],'exp_conf.yaml')) # remove
    trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
    
    # if running training config, then use the training config for testing
    tstng_exp_conf = trng_exp_conf
    tstng_exp_conf.update(tstng_conf)

    # update the testing experiment config file
    tstng_exp_conf['sim_params']['render'] = args.render_onscreen
    tstng_exp_conf['visualize_reference']  = args.visualize_reference
    tstng_exp_conf['visualize_oracle']  = args.visualize_oracle

    if exists_not_none('frame_recorder',tstng_exp_conf):
        if '.mp4' not in tstng_exp_conf['frame_recorder']['export_path']:
            tstng_conf['frame_recorder']['export_path'] = os.path.join(
            tstng_conf['frame_recorder']['export_path'],
            tstng_conf['exp_log_path'].replace("./logs/","")
            )
    
    if exists_not_none('export_logger',tstng_exp_conf):
        tstng_conf['export_logger']['export_path'] = os.path.join(
        tstng_conf['export_logger']['export_path'],
        tstng_conf['exp_log_path'].replace("./logs/","")
        )
    tstng_exp_conf_file =  open(tstng_exp_conf_path,'w')
    yaml.dump(tstng_exp_conf,tstng_exp_conf_file,default_flow_style=False,sort_keys=False)

    # create the environment
    env = env_factory(tstng_exp_conf_path)()
    env.sim.set_default_camera()

    # load the policy
    print("\ntesting experiment:",tstng_conf['exp_log_path'])
    policy = torch.load(os.path.join(tstng_conf['exp_log_path'],'actor.pt'))


    # run n episodes
    with torch.no_grad():
        for n_epi in range(tstng_conf['n_episodes']):
            if env.sim.viewer_paused:
                print('viewer paused, press space to unpause')
            terminate_and_exit = rollout_policy(
                                        env, 
                                        policy, 
                                        sync_camera=args.sync_camera,
                                        )
            if terminate_and_exit:
                break
        env.close()