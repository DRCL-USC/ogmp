import sys
sys.path.append("./")
from tools.plots.conf_for_paper import *

import numpy as np
import os
import argparse 
from tqdm import tqdm

# parameters
SS_DIST_MAX_COLOR = 'gray'
TRAJ_LINE_WIDTH = 0.5
OG_DATA_WIDTH_SCALE = 3
MARKER_SIZE = 20
QPOS2USE = [
                0,1,2,          # base_pos
                # 3,4,5,6,        # base_quat
                # 7,8,9,10,11,    # jpos_left
                # 12,13,14,15,16, # jpos_right
            ]
QVEL2USE =  [
                0,1,2,          # base_tvel
                # 3,4,5,          # base_avel
                # 6,7,8,9,10,     # jvel_left
                # 11,12,13,14,15, # jvel_right
            ]
CLUSTERCOLORS = [
                # 'C0', # blue reserved for elbow plot
                # 'C1',  # orange reserved for silhoutte plot
                 'C2',
                #  'C3', # red reserved for z space
                 'C4',
                 'C5',
                #  'C6', # i dont like this color
                #  'C7', # gray reserved for search space
                 'C8',
                 'C9'
                ]

# util funcs
# def ccworder(A):
#     A= A - np.mean(A, 1)[:, None]
#     return np.argsort(np.arctan2(A[1, :], A[0, :]))
def downsample(data, ds_rate=10):
    return data[::ds_rate,:]

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--path2logs", default="", type=str)  
parser.add_argument("--env_dt", default=0.03, type=float)
parser.add_argument("--data_dt", default=0.0005, type=float)  
args = parser.parse_args()

plot_path = args.path2logs+'/plots/'
os.makedirs(plot_path,exist_ok=True)

# latent mode points encountered and og log
lme = np.load(args.path2logs+'/lme.npy')
og_log = np.load(args.path2logs+'/w_oracle/log.npz') 
transition_occurs_at = None


og_lmp = og_log['mode_latent'][-1,:]
robot_qpos = og_log['qpos'][:,QPOS2USE]
robot_qvel = og_log['qvel'][:,QVEL2USE]
robot_state = np.hstack((robot_qpos,robot_qvel))
og_robot_traj = downsample(robot_state,ds_rate=int(args.env_dt/args.data_dt))
args.path2logs = args.path2logs+'/logs/'

logs =[obj for obj in  os.listdir(args.path2logs) if os.path.isdir(os.path.join(args.path2logs,obj))]
# print('logs: ',logs)
logs = sorted(logs,key=lambda x: int(x.split('.')[0]))





robot_trajs = []
latent_mode_set_rollouts = []
returns = []
epi_len = []
transition_occurs_at = None

# load rollout trajs
for i in tqdm(range(len(logs))):
    path2log =os.path.join(
                            args.path2logs,
                            str(i)+'/log.npz'
                            )
    log = np.load(path2log)
    # select the state variables to use
    robot_qpos = log['qpos'][:,QPOS2USE]
    robot_qvel = log['qvel'][:,QVEL2USE]
    robot_state = np.hstack((robot_qpos,robot_qvel))
    robot_state = downsample(robot_state,ds_rate=int(args.env_dt/args.data_dt))
    robot_trajs.append(robot_state)

    latent_mode_set_rollouts.append(log['mode_latent'][-1,:])

    if transition_occurs_at is None:
        for t in range(1,len(robot_state)):
            if robot_state[t,2] != og_robot_traj[t,2]:
                transition_occurs_at = t
                break

    
    path2mlog = os.path.join(
                            args.path2logs,
                            str(i)+'/metrics_log.npz'
                            )

    # metric log
    mlog = np.load(path2mlog)
    returns.append(mlog['returns'])
    epi_len.append(mlog['epi_len'])



# convert to numpy
robot_trajs = np.array(robot_trajs)
latent_mode_set_rollouts = np.array(latent_mode_set_rollouts)



eps = 0.1
min_x = min(latent_mode_set_rollouts[:,0]) - eps
max_x = max(latent_mode_set_rollouts[:,0]) + eps
min_y = min(latent_mode_set_rollouts[:,1]) - eps
max_y = max(latent_mode_set_rollouts[:,1]) + eps

# figure
fig0, ax2d = plt.subplots(1,2,figsize=(8,4))

in_z_space = []
for i,point in enumerate(latent_mode_set_rollouts):
    if robot_trajs[i,-1,2] <= 0.4:
        in_z_space.append(0)
    else:
        in_z_space.append(1)

# centre and max radis of the latent space
lme_center = np.mean(lme,axis=0)

lme_dist_max_at = np.argmax(np.linalg.norm(lme-lme_center,axis=1))
lme_dist_max = np.linalg.norm(lme[lme_dist_max_at]-lme_center)

n_point_in_id = 0
n_point_in_od = 0

n_point_in_id_in_z = 0
n_point_in_od_in_z = 0

# plot1,2: z space plot, state space plot
for is_in_z_space,z,x_traj in zip(in_z_space,latent_mode_set_rollouts,robot_trajs):
    


    if np.linalg.norm(z-lme_center) < lme_dist_max:
        n_point_in_id += 1
    else:
        n_point_in_od += 1
    
    # max alphas max_z and 0 alpha at min_z
    if is_in_z_space == 0:
        color = 'black'
        alpha_val = 0.25

    else:
        color = 'C3'
        alpha_val = 1.0 

        if np.linalg.norm(z-lme_center) < lme_dist_max:
            n_point_in_id_in_z += 1
        else:
            n_point_in_od_in_z += 1  
    ax2d[0].scatter(
                    z[0],
                    z[1],
                    c=color,
                    marker='o',
                    alpha= alpha_val,
                    s=MARKER_SIZE,
                    )
    ax2d[1].plot(
                    # x[:,0],
                    x_traj[:,2],
                    color=color,
                    linestyle='-',
                    linewidth=TRAJ_LINE_WIDTH,
                    alpha = alpha_val,
                    )

print('n_point_in_id: ',n_point_in_id,' n_point_in_od: ',n_point_in_od)
print('n_point_in_id_in_z: ',n_point_in_id_in_z,' n_point_in_od_in_z: ',n_point_in_od_in_z)

print('nu_ID:',n_point_in_id_in_z/n_point_in_id)
print('nu_OD:',n_point_in_od_in_z/n_point_in_od)

# plot the og_lmp and og_robot_traj
ax2d[0].scatter(
                og_lmp[0],
                og_lmp[1],
                c='C0',
                marker='o',
                alpha= 1.0,
                s=OG_DATA_WIDTH_SCALE*MARKER_SIZE,
                )
# annotate the og_lmp
ax2d[0].annotate(
                r'$z^{*}_{1}$',
                xy=(og_lmp[0],og_lmp[1]),
                xytext=(og_lmp[0]-1.0,og_lmp[1]+1.4),
                arrowprops=dict(
                                facecolor='black',
                                edgecolor='black',
                                arrowstyle='->',
                                ),
                )
ax2d[1].plot(
                # x[:,0],
                og_robot_traj[:,2],
                color='C0',
                linestyle='-',
                linewidth=OG_DATA_WIDTH_SCALE*TRAJ_LINE_WIDTH,
                alpha = 1.0,
                )
# plot a vline at transition_occurs_at
ax2d[1].axvline(transition_occurs_at,color='k',linestyle='--')

boundary_points = [
                    [min_x+eps,min_y+eps],
                    [max_x-eps,min_y+eps],
                    [max_x-eps,max_y-eps],
                    [min_x+eps,max_y-eps],
                    [min_x+eps,min_y+eps],
                ]
                            

boundary_points = np.array(boundary_points)

ax = ax2d[0]
ax.set_xlabel('latent axis 1')
ax.set_ylabel('latent axis 2')

ax.set_xlim(min_x,max_x)
ax.set_ylim(min_y,max_y)
# ax.grid()
ax.plot(
        boundary_points[:,0],
        boundary_points[:,1],
        c=SS_DIST_MAX_COLOR,
        linestyle='--',
        )


# for state space plots
ax2d[1].set_xlabel('timesteps')
ax2d[1].set_ylabel('base height')
ax2d[1].grid()

fig0.tight_layout()


# fig0.subplots_adjust(top=0.9)
# So far, nothing special except the managed prop_cycle. Now the trick:
# lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# Finally, the legend (that maybe you'll customize differently)
# fig0.legend(lines, labels, loc='upper center', ncol=9)

# handles, labels = ax2d[-1].get_legend_handles_labels()
# fig0.legend(handles, labels, loc='upper center')


fig0.savefig(plot_path+'transition_z_x_spaces.png')

plt.show()

exit()
