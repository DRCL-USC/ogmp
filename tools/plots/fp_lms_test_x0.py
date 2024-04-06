import numpy as np
import os
import argparse 
from tqdm import tqdm
import sys
sys.path.append("./")
from tools.plots.conf_for_paper import *



# parameters
ID_COLOR = 'deepskyblue'
LME_CENTER_COLOR = 'deepskyblue'
LME_DIST_MAX_COLOR = 'deepskyblue'
SS_DIST_MAX_COLOR = 'gray'
SS_ALPHA = 0.15
N_CLUSTERS = 4
TRAJ_LINE_WIDTH = 0.5
Z_SPACE_PROP_LINE_WIDTH = 2.5
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

# latent mode points encountered
lme = np.load(args.path2logs+'/lme.npy')
args.path2logs = args.path2logs+'/logs/'

logs =[obj for obj in  os.listdir(args.path2logs) if os.path.isdir(os.path.join(args.path2logs,obj))]
# print('logs: ',logs)
logs = sorted(logs,key=lambda x: int(x.split('.')[0]))





robot_trajs = []
latent_mode_set_rollouts = []
returns = []
epi_len = []

# load rollout trajs
min_len = np.inf
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
    
    # latent_mode_set_rollouts.append(log['mode_latent'][0,:])

    latent_mode_set_rollouts.append(log['mode_latent'][-1,:])

    min_len = min(min_len,robot_state.shape[0])
    robot_trajs.append(robot_state)
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
fig0, ax2d = plt.subplots(1,5,figsize=(20,5))

# plot0: lme paramters, span points randomply downsample and plot
n_lme_points = int(0.5*len(lme) )

ax2d[0].scatter(
                lme[0,0],
                lme[0,1],
                c=ID_COLOR,
                marker='x',
                alpha=0.5,
                s=MARKER_SIZE,
                label=r'$z$ points $ID$',
                )
for i in range(1, n_lme_points):    
    ax2d[0].scatter(
                    lme[i,0],
                    lme[i,1],
                    c=ID_COLOR,
                    marker='x',
                    alpha=0.25,
                    s=MARKER_SIZE,
                    )
# ax2d[0].scatter(
#                 lme[:,0],
#                 lme[:,1],
#                 c=ID_COLOR,
#                 marker='x',
#                 alpha=0.25,
#                 s=MARKER_SIZE,
#                 )


ax2d[0].scatter(
                latent_mode_set_rollouts[:,0],
                latent_mode_set_rollouts[:,1],
                c='black',
                marker='o',
                alpha=SS_ALPHA,
                s=MARKER_SIZE,
                
                )



lme_center = np.mean(lme,axis=0)
# plot the search space radius

ss_midpoint = (lme_center + np.array([max_x-eps,lme_center[1]]))/2
# annotate the search space radius with an arrow

ax2d[0].text(
                ss_midpoint[0],
                ss_midpoint[1]-1.3,
                r'$SS$',
                # xy=ss_midpoint,
                # arrowprops=dict(
                #                 facecolor='black',
                #                 edgecolor='black',
                #                 arrowstyle='->',
                #                 ),
                )
ax2d[0].scatter(
                lme_center[0],
                lme_center[1],
                c='black',#LME_CENTER_COLOR,
                marker='o',
                alpha=1.0,
                s=3*MARKER_SIZE,
                )
ax2d[0].annotate(
                r'$a_{ID}$',
                xy=lme_center,
                xytext=(lme_center[0]-0.5,lme_center[1]+1.0),
                arrowprops=dict(
                                facecolor='black',
                                edgecolor='black',
                                arrowstyle='->',
                                ),

                )

lme_dist_max_at = np.argmax(np.linalg.norm(lme-lme_center,axis=1))
lme_dist_max = np.linalg.norm(lme[lme_dist_max_at]-lme_center)
# plt the line from lme_center to line_dist_max_at
ax2d[0].plot(
            [lme_center[0],lme[lme_dist_max_at,0]],
            [lme_center[1],lme[lme_dist_max_at,1]],
            c= 'black', #LME_DIST_MAX_COLOR,
            linestyle='--',
            alpha=1.0,
            linewidth=Z_SPACE_PROP_LINE_WIDTH,
            )
# annotate the lme_dist_max with an arrow 
lme_dist_max_midpoint = (lme_center + lme[lme_dist_max_at])/2
ax2d[0].annotate(
                r'$d_{ID}$',
                xy=lme_dist_max_midpoint,
                xytext=(lme_center[0]- 1.0,lme_center[1]-1.0),
                arrowprops=dict(
                                facecolor='black',
                                edgecolor='black',
                                arrowstyle='->',
                                ),
                )


# daw a circle at lme_center with radius lme_dist_max
ax2d[0].add_artist(plt.Circle(
                                lme_center,
                                lme_dist_max,
                                color=ID_COLOR,
                                fill=False,
                                linestyle='--',
                                alpha=1.0,
                                linewidth=Z_SPACE_PROP_LINE_WIDTH,
                                label=r'circumcircle of $ID$',
                            ))







in_z_space = []
for i,point in enumerate(latent_mode_set_rollouts):
    if robot_trajs[i,-1,2] <= 0.4:
        in_z_space.append(0)
    else:
        in_z_space.append(1)

# plot1: z space plot
is_first_z_point = True
is_first_s_point = True


n_point_in_id = 0
n_point_in_od = 0

n_point_in_id_in_z = 0
n_point_in_od_in_z = 0

for i,point in enumerate(latent_mode_set_rollouts):
    
    # max alphas max_z and 0 alpha at min_z

    if np.linalg.norm(point-lme_center) < lme_dist_max:
        n_point_in_id += 1
    else:
        n_point_in_od += 1

    if in_z_space[i] == 0:
        color = 'black'
        alpha_val = SS_ALPHA
        label = 'Search Space ('+r'$SS$)' if is_first_s_point else None
        is_first_s_point = False


        
    else:
        color = 'C3'
        alpha_val = 1.0 
        label = r'$\mathcal{Z}(x)$' if is_first_z_point else None
        is_first_z_point = False
        if np.linalg.norm(point-lme_center) < lme_dist_max:
            n_point_in_id_in_z += 1
        else:
            n_point_in_od_in_z += 1


    ax2d[1].scatter(
            point[0],
            point[1],
            c=color,
            marker='o',
            alpha= alpha_val,
            s=MARKER_SIZE,
            label=label,
            ) 

print('n_point_in_id: ',n_point_in_id,' n_point_in_od: ',n_point_in_od)
print('n_point_in_id_in_z: ',n_point_in_id_in_z,' n_point_in_od_in_z: ',n_point_in_od_in_z)

print('nu_ID:',n_point_in_id_in_z/n_point_in_id)
print('nu_OD:',n_point_in_od_in_z/n_point_in_od)

# search space and lme points
ax2d[1].add_artist((plt.Circle(
                                lme_center,
                                lme_dist_max,
                                color=ID_COLOR,
                                fill=False,
                                linestyle='--',
                                alpha=1.0,
                                linewidth=Z_SPACE_PROP_LINE_WIDTH,
                            ))
                    )   

# plt the points in search space but not in z space
for i,point in enumerate(latent_mode_set_rollouts):
    if in_z_space[i] == 0:
        ax2d[3].scatter(
                        point[0],
                        point[1],
                        c='black',
                        marker='o',
                        alpha=SS_ALPHA,
                        s=MARKER_SIZE,
                        )        
    # ax2d[3].annotate(
    #                 i,
    #                 xy=point,
    #                 xytext=(point[0],point[1]),
    #                 fontsize=10.0,
    #                 )
robot_trajs_in_z_space = [robot_traj for i,robot_traj in enumerate(robot_trajs) if in_z_space[i]==1]
latent_mode_set_rollouts_in_z_space = [latent_mode_set_rollout for i,latent_mode_set_rollout in enumerate(latent_mode_set_rollouts) if in_z_space[i]==1]
# only consider the z space trajectories
robot_trajs = np.array(robot_trajs_in_z_space)
latent_mode_set_rollouts = np.array(latent_mode_set_rollouts_in_z_space)

# normalise data before kmeans
state_max = np.max(np.max(np.abs(robot_trajs),axis=0),axis=0)
robot_trajs = robot_trajs/state_max
print('norm. traj dataset shape: ',robot_trajs.shape)

# make a trjectory(2d) in to a vector(1d)
pixel_data = robot_trajs.reshape(robot_trajs.shape[0],-1)

# kmeans'
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull

# make an elbow plot (main axis) and silhoutte score(twin axis) for n_clusters from 1 to 15
n_cluster_range = np.arange(2,min(15,len(pixel_data)))
elbow_scores = []
silhoutte_scores = []

for n_clusters in n_cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init="auto").fit(pixel_data)
    elbow_scores.append(kmeans.inertia_)
    silhoutte_scores.append(silhouette_score(pixel_data, kmeans.labels_))


# from silhoutte score plot, choose the n_clusters with max silhoutte score
optimal_n_clusters = n_cluster_range[np.argmax(silhoutte_scores)]


N_CLUSTERS = 4 #optimal_n_clusters

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42,n_init="auto").fit(pixel_data)
mode_clusters = [[] for _ in range(N_CLUSTERS)]

for i,label in enumerate(kmeans.labels_): 
    mode_clusters[label].append(i)

# smallest cluster comes first
mode_clusters.sort(key=len)

# normalize elbow scores
elbow_scores = (elbow_scores- np.min(elbow_scores))/(np.max(elbow_scores) - np.min(elbow_scores))
elbow_scores = elbow_scores.round(2)
# normalize silhoutte scores
silhoutte_scores = (silhoutte_scores- np.min(silhoutte_scores))/(np.max(silhoutte_scores) - np.min(silhoutte_scores))
silhoutte_scores = silhoutte_scores.round(2)

# plot the elbow and silhoutte scores
ax2d[2].plot(elbow_scores, 'C0', label='elbow ')
ax2d[2].plot(silhoutte_scores, 'C1', label='silhoutte')
ax2d[2].set_xticks(
            np.arange(len(n_cluster_range)),
            n_cluster_range,
            rotation=90
            )
# plot a vline at N_CLUSTERS
ax2d[2].axvline(
                    x=np.where(n_cluster_range==N_CLUSTERS)[0][0], 
                    color='black', 
                    linestyle='--'
                )


# plot the silhoutte scores
for ci in range(N_CLUSTERS):
    points_this_cluster = latent_mode_set_rollouts[mode_clusters[ci],:]
    print('cluster',ci,'shape: ',points_this_cluster.shape)

    ax2d[3].scatter(
                points_this_cluster[:,0], 
                points_this_cluster[:,1], 
                c=CLUSTERCOLORS[ci],
                s=MARKER_SIZE,
                marker='o',
                label='mode '+str(ci+1),
                )  

    # print state trajectories
    for case in mode_clusters[ci][1:]:
        ax2d[4].plot(
                        # robot_trajs[case,:,0],
                        robot_trajs[case,:,2],
                        color=CLUSTERCOLORS[ci],
                        linestyle='-',
                        linewidth=TRAJ_LINE_WIDTH,
                        alpha = 0.5,
                        )

ax2d[3].add_artist((plt.Circle(
                                lme_center,
                                lme_dist_max,
                                color=ID_COLOR,
                                fill=False,
                                linestyle='--',
                                alpha=1.0,
                                linewidth=Z_SPACE_PROP_LINE_WIDTH,
                            ))
                    )   

boundary_points = [
                    [min_x+eps,min_y+eps],
                    [max_x-eps,min_y+eps],
                    [max_x-eps,max_y-eps],
                    [min_x+eps,max_y-eps],
                    [min_x+eps,min_y+eps],
                ]
                            

boundary_points = np.array(boundary_points)

# for mode space plots
for ax_i in [0,1,3]:
    ax = ax2d[ax_i]
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

# for metric plots
ax2d[2].set_xlabel('no. of clusters')
ax2d[2].set_ylabel('normalized score')
ax2d[2].grid()

# for state space plots
ax2d[4].set_xlabel('timesteps')
ax2d[4].set_ylabel('base height')
ax2d[4].grid()

fig0.tight_layout()


fig0.subplots_adjust(top=0.75)
# So far, nothing special except the managed prop_cycle. Now the trick:
lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# Finally, the legend (that maybe you'll customize differently)
fig0.legend(lines, labels, loc='upper center', ncol=5)

# handles, labels = ax2d[-1].get_legend_handles_labels()
# fig0.legend(handles, labels, loc='upper center')


fig0.savefig(plot_path+'kmeans_'+str(N_CLUSTERS)+'_2d_all.png')

plt.show()

exit()
