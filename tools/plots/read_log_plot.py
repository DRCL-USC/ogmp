import numpy as np
import numpy.fft as fft

import matplotlib.pyplot as plt
import yaml
from dm_control.utils import transformations
import sys

import argparse 
sys.path.append("./")

from src.misc_funcs import custom_tight_layout

SIMRATE = 30
ACTION_RAMP_EXPLICIT = False


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="./log.npz", type=str)  
parser.add_argument("--model_prop_path", default="./dtsd/envs/rsc/models/mini_biped/xmls/biped_simple.yaml", type=str)
parser.add_argument("--mode_space_path", default="dtsd/analysis_results/fgb_fx0_sρ_ae32/prtm3_vx1/dim_2/tests/trng_sanity/mode_space.npz", type=str)
parser.add_argument("--robot_name", default="drcl_biped", type=str) 
parser.add_argument("--sim_dt", default=5e-4, type=float)  
parser.add_argument("--ti", default= 0, type=int)
parser.add_argument("--tf", default=-1, type=int)


args = parser.parse_args()

data = np.load(args.datapath)
prop_file = open(args.model_prop_path) 
model_prop = yaml.load(prop_file, Loader=yaml.FullLoader)    

print("log summary:")
for file in data.files:
    print("\t",file,data[file].shape)

if args.tf == -1:
    args.tf = data[file].shape[0]



plots2make = [

                # 'base_states',
                # 'act',
                # 'base_acc',
                'jpos_action_tau',
                # 'jpos_action_fft',
                # 'jvel',
                # 'obs',
                # "mode_latent",
                # "mode_latent_2danim",
                # 'joint_power',
                # 'fraude'
                
]

if 'base_states' in plots2make:
    # figure 1: base_states
    fig, axs = plt.subplots(nrows=2,ncols=6)
    fig.set_size_inches(20, 12)

    fig.suptitle("base_states")
    base_rpy = []
    for qpos in data['qpos']:
        quat = qpos[
                        model_prop[args.robot_name]['ids']['base_ori'][0]:
                        model_prop[args.robot_name]['ids']['base_ori'][-1]+1
                    ]
        b_rpy = transformations.quat_to_euler(quat=quat)
        base_rpy.append(b_rpy)


    base_rpy = np.array(base_rpy)
    # print(base_rpy.shape)
    custom_tight_layout()


    for i,(bpi,btvi,bavi) in enumerate(zip(
                                            model_prop[args.robot_name]['ids']['base_pos'],
                                            model_prop[args.robot_name]['ids']['base_tvel'],
                                            model_prop[args.robot_name]['ids']['base_avel'],

                                            )
                                    ):
        axs[0,bpi].plot(data["qpos"][args.ti:args.tf,bpi])
        axs[0,3+bpi].plot(base_rpy[args.ti:args.tf,i])
        avg_rpy = np.mean(base_rpy[args.ti:args.tf,i])
        axs[0,3+bpi].hlines(avg_rpy,args.ti,args.tf,color='r',linestyle='--')
        axs[0,3+bpi].annotate(
                                str(np.round(avg_rpy,2)),
                                xy=(args.tf, avg_rpy),
                                rotation=0,
                                color='r',
                            )


        axs[1,btvi].plot(data["qvel"][args.ti:args.tf,btvi])
        # avg tvel
        avg_tvel = np.mean(data["qvel"][args.ti:args.tf,btvi])
        axs[1,btvi].hlines(avg_tvel,args.ti,args.tf,color='r',linestyle='--')
        axs[1,btvi].annotate( 
                                str(np.round(avg_tvel,2)),
                                xy=(args.tf, avg_tvel),
                                rotation=0,
                                color='r',
                            )

        axs[1,bavi].plot(data["qvel"][args.ti:args.tf,bavi])
        # avg avel
        avg_avel = np.mean(data["qvel"][args.ti:args.tf,bavi])
        axs[1,bavi].hlines(avg_avel,args.ti,args.tf,color='r',linestyle='--')
        axs[1,bavi].annotate( 
                                str(np.round(avg_avel,2)),
                                xy=(args.tf, avg_avel),
                                rotation=0,
                                color='r',
                            )
    for ax_row in axs:
        for ax in ax_row:
            ax.grid()
            # ax.legend()

    plt.savefig(args.datapath.replace('log.npz','base_states.png'))

if 'base_acc' in plots2make:

    fig, axs = plt.subplots(nrows=1,ncols=6)
    fig.suptitle("base_acc")
    custom_tight_layout()
    fig.set_size_inches(20, 12)
    for i,(btvi,bavi) in enumerate(zip(
                                            model_prop[args.robot_name]['ids']['base_tvel'],
                                            model_prop[args.robot_name]['ids']['base_avel'],

                                            )
                                    ):    
        axs[btvi].plot(data["qacc"][args.ti:args.tf,btvi])
        axs[bavi].plot(data["qacc"][args.ti:args.tf,bavi])

    for ax in axs:
        ax.grid()
        # ax.legend()

    plt.savefig(args.datapath.replace('log.npz','base_acc.png'))

if 'fraude' in plots2make:

    fig, axs = plt.subplots(nrows=1,ncols=1)
    fig.suptitle("base_acc")
    custom_tight_layout()
    fig.set_size_inches(20, 12)
    
    leg_length = 0.459
    vx = data["qvel"][args.ti:args.tf,model_prop[args.robot_name]['ids']['base_tvel'][0]]
    fraude = np.power(vx,2)/(9.81*leg_length)
    axs.plot(fraude)
    print(' fraude mean',np.max(fraude))
    axs.grid()

    plt.savefig(args.datapath.replace('log.npz','fraude.png'))

if 'jpos_action_tau' in plots2make:
    
    
    # figure 2: joint_pos, actions and ctrl
    if args.robot_name == 'drcl_biped':
        n_actuate_dof = 5
    elif args.robot_name == 'hector_biped_v1':
        n_actuate_dof = 5
    


    fig, axs = plt.subplots(nrows=2,ncols=n_actuate_dof)
    fig.set_size_inches(20, 12)

    fig.suptitle("jpos, actions, and tau")
    for i,jpi in enumerate(model_prop[args.robot_name]['ids']['jpos']):
        
        r = i // n_actuate_dof
        c = i % n_actuate_dof
        # print(r,c)
        # axs[r,c].axhline(y=np.degrees(MOTOR_LIMITS[i,0]),color='orange',linestyle='-')
        # axs[r,c].axhline(y=np.degrees(MOTOR_LIMITS[i,1]),color='orange',linestyle='-')

        q_max = np.max(data['qpos'][:,jpi])
        q_max_at = np.where(data['qpos'][:,jpi] == q_max)[0][0]
        # axs[r,c].scatter(
        #                     q_max_at,
        #                     np.degrees(q_max),
        #                     color='b',
        #                     marker='x',
        #                     # label='max_min'
        #                 )
        # axs[r,c].annotate(
                    # "M:"+str(np.round(np.degrees(q_max),1))+' deg',
                    # xy=(q_max_at, np.degrees(q_max)+2), 
                    # rotation=0,

                    # )
        q_min = np.min(data['qpos'][:,jpi])
        q_min_at = np.where(data['qpos'][:,jpi] == q_min)[0][0]
        # axs[r,c].scatter(
        #                     q_min_at,
        #                     np.degrees(q_min),
        #                     color='b',
        #                     marker='x',
        #                     # label='max_min'
        #                 )
        # axs[r,c].annotate(
        #             "m:"+str(np.round(np.degrees(q_min),1))+' deg',
        #             xy=(q_min_at, np.degrees(q_min)-2), 
        #             rotation=0,

        #             )


        # original joint pos

        axs[r,c].plot(
                        # np.degrees(
                            data['qpos'][:,jpi]
                            # )
                        )

        axs[r,c].plot(
                        # np.degrees(
                            data['action'][:,i]
                            # )
                            ,
                            'r--',alpha=0.5
                    )
        
        # twin_ax = axs[r,c].twinx()
        

        # ax_xtwin = axs[r,c].twinx()
        # ax_xtwin.plot(
        #                 data['actuator_force'][:,i],'g--',alpha=0.5
        #             )

        # tau_max = np.max(data['actuator_force'][:,i])
        # tau_max_at = np.where(data['actuator_force'][:,i] == tau_max)[0][0]
        # ax_xtwin.scatter(
        #                     tau_max_at,
        #                     tau_max,
        #                     color='g',
        #                     marker='x',
        #                     # label='max_min'
        #                 )
        # ax_xtwin.annotate(
        #             "M:"+str(np.round(tau_max,1))+' Nm',
        #             xy=(tau_max_at, tau_max), 
        #             rotation=0,

        #             )

        # tau_min = np.min(data['actuator_force'][:,i])
        # tau_min_at = np.where(data['actuator_force'][:,i] == tau_min)[0][0]
        # ax_xtwin.scatter(
        #                     tau_min_at,
        #                     tau_min,
        #                     color='g',
        #                     marker='x',
        #                     # label='max_min',

        #                 )
        # ax_xtwin.annotate(
        #             "m:"+str(np.round(tau_min,1))+' Nm',
        #             xy=(tau_min_at, tau_min), 
        #             rotation=0,
        #             )

        # ax_xtwin.tick_params(
        #                     axis='y',
        #                     # reset=True,
        #                     labelrotation=90,
        #                     # direction='in',
        #                 ) 


        if c == 0:
            axs[r,c].set_ylabel("jpos ")
        elif c == 4:
            axs[r,c].set_ylabel("action ")


    for ax_row in axs:
        for ax in ax_row:
            ax.grid()
            ax.tick_params(
                                axis='y',
                                # reset=True,
                                labelrotation=90,
                                # direction='in',
                            )    
            # ax.legend()
    custom_tight_layout()
    plt.savefig(args.datapath.replace('log.npz','jpos_action.png'))

if 'jvel' in plots2make:
    # figure 3: joint_vel
    fig, axs = plt.subplots(nrows=2,ncols=5)
    fig.set_size_inches(20, 12)

    fig.suptitle("jvel")
    for i,jvi in enumerate(model_prop[args.robot_name]['ids']['jvel']):
        r = i // 5
        c = i % 5
        axs[r,c].plot(data['qvel'][:,jvi])
    for ax_row in axs:
        for ax in ax_row:
            ax.grid()
            # ax.legend()
    
    custom_tight_layout()
    plt.savefig(args.datapath.replace('log.npz','jvel.png'))

if 'jpos_action_fft' in plots2make:
    
    # figure 3: joint_pos, actions and ctrl
    n_actuate_dof = 5
    fig, axs = plt.subplots(nrows=2,ncols=n_actuate_dof)
    fig.set_size_inches(20, 12)

    fig.suptitle("jpos, actions fft")
    for i,jpi in enumerate(model_prop[args.robot_name]['ids']['jpos']):
        
        r = i // n_actuate_dof
        c = i % n_actuate_dof

        # original joint pos
        act = data['action'][:,i]

        act_fft = fft.fft(act)
        freq = fft.fftfreq(len(act),d=args.sim_dt)

        axs[r,c].plot(
                        freq,
                        np.abs(act_fft),
                        label='act'
                    )
        # jpos = data['qpos'][:,jpi]
        # jpos_fft = fft.fft(jpos)
        # axs[r,c].plot(
        #                 freq,
        #                 np.abs(jpos_fft),
        #                 label='jpos'
        #             )
        # axs[r,c].set_yscale('log')
        # axs[r,c].set_ylim(1, 1e3)
        #  find the cuttoff freq, i.e at magnitude = 0.5*max(magnitude)

        max_amp = np.max(np.abs(act_fft))

        # bandwidth = 0.2*max_amp
        # freqs_in_bandwidth = freq[np.abs(act_fft) > bandwidth]
        # print("bandwidth:",freqs_in_bandwidth)

        # draww hlines at 0.05, 0.1, 0.2, 0.2
        for p in [0.05,0.1,0.2,0.5]:
            axs[r,c].axhline(
                            p*max_amp,
                            color='r',
                            linestyle='--',
                            label=str(p)
                            )


        # mark the cutoff freq
        # axs[r,c].axvline(cutoff_freq,color='r',linestyle='--',label='cutoff_freq')
        # mar the magnitude at cutoff freq as a horizontal line

        
        axs[r,c].set_xlabel("freq")
        axs[r,c].set_ylabel("amp (log)")
        axs[r,c].legend()
        axs[r,c].grid()
    custom_tight_layout()
    plt.savefig(args.datapath.replace('log.npz','jpos_action_fft.png'))

if 'obs' in plots2make:



    obs_labels = [
                    'base_pos_z',
                    'base_ori_w',
                    'base_ori_x',
                    'base_ori_y',
                    'base_ori_z',
                    'l_jpos_hy',
                    'l_jpos_hr',
                    'l_jpos_thigh',
                    'l_jpos_knee',
                    'l_jpos_ankle',

                    'r_jpos_hy',
                    'r_jpos_hr',
                    'r_jpos_thigh',
                    'r_jpos_knee',
                    'r_jpos_ankle',

                    'base_tvel_x',
                    'base_tvel_y',
                    'base_tvel_z',

                    'base_avel_roll',
                    'base_avel_pitch',
                    'base_avel_yaw',

                    'l_jvel_hy',
                    'l_jvel_hr',
                    'l_jvel_thigh',
                    'l_jvel_knee',
                    'l_jvel_ankle',

                    'r_jvel_hy',
                    'r_jvel_hr',
                    'r_jvel_thigh',
                    'r_jvel_knee',
                    'r_jvel_ankle',
                ]
    # figure 4: obs
    obs_dim = data['observation'].shape[1]
    total_plots = 36
    if obs_dim > total_plots:
        print("obs_dim > "+str(total_plots)+", last n rows are not plotted")
    
    n_rows = 4
    n_cols = int(total_plots/n_rows)
    fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols)
    fig.set_size_inches(20, 12)

    fig.suptitle("obs")
    
    
    for i in range(min(obs_dim,total_plots)):
        r = i // n_cols
        c = i % n_cols
        axs[r,c].plot(
                        np.arange(args.ti,args.tf),
                        data['observation'][args.ti:args.tf,i]
                    )
        try:
            axs[r,c].set_title(obs_labels[i])
        except:
            axs[r,c].set_title("obs_"+str(i))
   
    
    
    for ax_row in axs:
        for ax in ax_row:
            ax.grid()
            ax.tick_params(
                                axis='y',
                                # reset=True,
                                labelrotation=90,
                                # direction='in',
                            )
            # ax.legend()
    plt.subplots_adjust(
                            left = 0.05,
                            right= 0.95,
                            bottom=0.05,
                            top=0.94,
                            hspace=0.245,
                            wspace=0.140,
                            )
    plt.savefig(args.datapath.replace('log.npz','obs.png'))

if 'act' in plots2make:
    # figure 5: act
    act_dim = data['action'].shape[1]    
    fig, axs = plt.subplots(nrows=2,ncols=5)
    fig.set_size_inches(20, 12)

    fig.suptitle("act")
    
    
    for i in range(act_dim):
        r = i // 5
        c = i % 5
        axs[r,c].plot(
                        np.arange(args.ti,args.tf),
                        data['action'][args.ti:args.tf,i]
                    )
        axs[r,c].set_title("act_"+str(i))
   
    
    
    for ax_row in axs:
        for ax in ax_row:
            ax.grid()
            ax.tick_params(
                                axis='y',
                                # reset=True,
                                labelrotation=90,
                                # direction='in',
                            )
            # ax.legend()
    plt.subplots_adjust(
                            left = 0.05,
                            right= 0.95,
                            bottom=0.05,
                            top=0.94,
                            hspace=0.245,
                            wspace=0.140,
                            )
    plt.savefig(args.datapath.replace('log.npz','act.png'))

if 'mode_latent' in plots2make:
    # figure 5: mode_latent
    mode_len = data['mode_latent'].shape[1]

    fig, axs = plt.subplots(nrows=mode_len,ncols=1)
    fig.set_size_inches(20, 6)
    fig.suptitle("mode_latent")
    
    for i in range(mode_len):
    
        axs[i].plot(
                    np.arange(args.ti,args.tf),
                    data['mode_latent'][args.ti:args.tf,i]
                    
                    )
        axs[i].grid()
        axs[i].set_ylabel("mode_axis_"+str(i))
    
    # axs.legend()
    # custom_tight_layout()
    plt.tight_layout
    plt.savefig(args.datapath.replace('log.npz','mode_latent.png'))

if 'mode_latent_2danim' in plots2make:
    
    
    
    lstm_me_mode_space = np.load(args.mode_space_path)
    
    # plt.figure(figsize=(20,12))

    # from traj data to account for ood modes

    x_min = lstm_me_mode_space['xlim'][0]
    x_max = lstm_me_mode_space['xlim'][1]

    y_min = lstm_me_mode_space['ylim'][0]
    y_max = lstm_me_mode_space['ylim'][1]

    if x_min > np.min(data['mode_latent'][:,0]):
        x_min = np.min(data['mode_latent'][:,0]) - 0.1
    if x_max < np.max(data['mode_latent'][:,0]):
        x_max = np.max(data['mode_latent'][:,0]) +0.1
    if y_min > np.min(data['mode_latent'][:,1]):
        y_min = np.min(data['mode_latent'][:,1]) - 0.1
    if y_max < np.max(data['mode_latent'][:,1]):
        y_max = np.max(data['mode_latent'][:,1]) + 0.1



    plt.xlim(
                x_min,
                x_max
            )
    plt.ylim(
                y_min,
                y_max
            )

    # print()
    for fi, file in enumerate(lstm_me_mode_space.files):
        if file not in ['xlim','ylim'] and lstm_me_mode_space[file].shape[0] > 0:
            plt.scatter(
                            lstm_me_mode_space[file][:,0],
                            lstm_me_mode_space[file][:,1],
                            label=file,
                            alpha=0.5,
                            color='C'+str(fi),
                        )
    plt.grid()
    plt.legend()
    plt.tight_layout()


    prev_x = -np.inf    #log['mode_latent'][0,0]
    prev_y = -np.inf    #log['mode_latent'][0,1]
    
    mode_k = 0
    last_mode_lasted_for = 0
    for (x,y) in zip(data['mode_latent'][:,0],data['mode_latent'][:,1]):
        
        if x != prev_x and y != prev_y:
            print("mode_"+str(mode_k),"lasted for",last_mode_lasted_for*args.sim_dt,"steps")
            plt.scatter(x,y,color='r')
            plt.annotate(
                            str(mode_k),
                            xy=(x+0.01,y), 
                            )
            plt.plot([prev_x,x],[prev_y,y],color='b')
            
            if last_mode_lasted_for > 0:
                plt.pause(last_mode_lasted_for*args.sim_dt)
            else:
                plt.pause(0.9)
            mode_k += 1
            last_mode_lasted_for = 0
        else:
            last_mode_lasted_for += 1

        prev_x = x
        prev_y = y

    plt.savefig(args.datapath.replace('log.npz','mode_latent_2danim.png'))
    plt.show()

if 'joint_power' in plots2make:
    jvel_ids = model_prop[args.robot_name]['ids']['jvel']
    
    # efficiency = lambda tau: -0.006*tau**6 + 0.0809*tau**5 - 0.415*tau**4 + 1.1*tau**3 - 1.7*tau**2 + 1.5*tau + 0.1
    efficiency = lambda tau: -0.006*tau**6 + 0.0809*tau**5 - 0.436*tau**4 + 1.2*tau**3 - 1.7924*tau**2 + 1.284*tau + 0.455
    efficiency2 = lambda tau: -0.0047*tau**6 + 0.0680*tau**5 -0.3881*tau**4 +  1.1214*tau**3 -1.7237*tau**2 +1.2576*tau + 0.4595
    

    joint_names = [
                    'l_hip_y',
                    'l_hip_r',
                    'l_thigh',
                    'l_knee',
                    'l_ankle',
                    'r_hip_y',
                    'r_hip_r',
                    'r_thigh',
                    'r_knee',
                    'r_ankle',
    ]
    # for i,jvi in enumerate(jvel_ids):
    

    #     mech_power = np.abs(data['actuator_force'][:,i]*data['qvel'][:,jvi])

    #     allowed_max_mech_power = efficiency(np.abs(data['actuator_force'][:,i])/9.1)*400 #data['actuator_force'][:,i]*data['qvel'][:,jvi]

    #     plt.title("joint_"+str(i)+': '+joint_names[i])
    #     plt.hlines(
    #                 y=400,
    #                 xmin=0,
    #                 xmax=mech_power.shape[0],
    #                 color='k',
    #                 linestyle='--',
    #                 label='ideal_power_max'
    #             )
    #     plt.plot(mech_power,'-',label='power')
    #     plt.plot(allowed_max_mech_power,'--',label='real_power_max')

    #     plt.legend()
    #     plt.grid()
    #     plt.tight_layout()        
    #     plt.show()
    #     plt.close()


    fig, axs = plt.subplots(nrows=2,ncols=5)
    fig.set_size_inches(20, 12)

    axs_flat = axs.flatten()
    for i,jvi in enumerate(jvel_ids):
    

        mech_power = np.abs(data['actuator_force'][:,i]*data['qvel'][:,jvi])

        allowed_max_mech_power = efficiency(np.abs(data['actuator_force'][:,i])/9.1)*400 #data['actuator_force'][:,i]*data['qvel'][:,jvi]

        axs_flat[i].set_title("joint_"+str(i)+': '+joint_names[i])
        axs_flat[i].hlines(
                    y=400,
                    xmin=0,
                    xmax=mech_power.shape[0],
                    color='k',
                    linestyle='--',
                    label='ideal_power_max'
                )
        axs_flat[i].plot(mech_power,'-',label='power')
        axs_flat[i].plot(allowed_max_mech_power,'--',label='real_power_max')
        

        axs_flat[i].grid()

    axs_flat[-1].legend()

    custom_tight_layout()
    plt.savefig(args.datapath.replace('log.npz','joint_power.png')) 

plt.show()

