import numpy as np
import matplotlib.pyplot as plt
import argparse 
from tqdm import tqdm
import os

QPOS2USE = [
                0,1,2,          # base_pos
                3,4,5,6,        # base_quat
                7,8,9,10,11,    # jpos_left
                12,13,14,15,16, # jpos_right
            ]
QVEL2USE =  [
                0,1,2,          # base_tvel
                3,4,5,          # base_avel
                6,7,8,9,10,     # jvel_left
                11,12,13,14,15, # jvel_right
            ]
def downsample(data, ds_rate=10):
    return data[::ds_rate,:]
parser = argparse.ArgumentParser()
parser.add_argument("--path2logs", default="", type=str)  
parser.add_argument("--path2modespace", default="logs/encoders/fgb_fx0_sρ_ae32/prtm3_vx1/dim_2/mode_space.npz", type=str)
parser.add_argument("--env_dt", default=0.03, type=float)
parser.add_argument("--data_dt", default=0.0005, type=float)  
args = parser.parse_args()

plot_path = args.path2logs+'/plots/'
# os.makedirs(plot_path,exist_ok=True)

args.path2logs = args.path2logs+'/logs/'
logs = sorted(os.listdir(args.path2logs),key=lambda x: int(x.split('.')[0]))

# load mode set induced by mode encoder
data = np.load(args.path2modespace)
mode_set_encoder = []
mode_set_encoder_label = []
mode_set_encoder_mean = []
for file in data.files:
    if file in ['flat','gap','block']:
        mode_set_encoder.append(data[file])
        mode_set_encoder_mean.append(np.mean(mode_set_encoder[-1],axis=0))
        mode_set_encoder_label.append(file)


robot_trajs = []
mode_latents = []
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
    robot_qpos = log['qpos'][:,QPOS2USE]
    robot_qvel = log['qvel'][:,QVEL2USE]
    robot_state = np.hstack((robot_qpos,robot_qvel))
    robot_state = downsample(
                                robot_state,
                                ds_rate=int(args.env_dt/args.data_dt)
                            )
    mode_latents.append(log['mode_latent'][0,:])    
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

reduce_trajs = []
for traj in robot_trajs:
    reduce_trajs.append(traj[:min_len,:])

# chop trajs to min_len
robot_trajs = reduce_trajs
print('min_len: ',min_len)
# convert to numpy
robot_trajs = np.array(robot_trajs)
mode_latents = np.array(mode_latents)

# normalise data before kmeans
state_max = np.max(np.max(np.abs(robot_trajs),axis=0),axis=0)
robot_trajs = robot_trajs/state_max
print('norm. traj dataset shape: ',robot_trajs.shape)

# make a trjectory(2d) in to a vector(1d)
pixel_data = robot_trajs.reshape(robot_trajs.shape[0],-1)
print('pixel_data shape: ',pixel_data.shape)

# manifold learning


# cluster the data
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(pixel_data)
high_dim_labels = kmeans.labels_


fig, axs = plt.subplots(2,5,figsize=(20,4))
from sklearn.manifold import *


# isomanp
iso = Isomap(n_components=2)
iso.fit(pixel_data)
iso_embedding = iso.transform(pixel_data)
# plot emmdeding based on labels
axs[0,0].scatter(iso_embedding[:,0],iso_embedding[:,1],c=high_dim_labels)
axs[0,0].set_title('isomap')
# cluster in low dim
iso_kmeans = KMeans(n_clusters=3, random_state=0).fit(iso_embedding)
iso_labels = iso_kmeans.labels_
axs[1,0].scatter(iso_embedding[:,0],iso_embedding[:,1],c=iso_labels)


# locally linear embedding
lle = LocallyLinearEmbedding(n_components=2)
lle_embedding = lle.fit_transform(pixel_data)
axs[0,1].scatter(lle_embedding[:,0],lle_embedding[:,1],c=high_dim_labels)
axs[0,1].set_title('lle')
# cluster in low dim
lle_kmeans = KMeans(n_clusters=3, random_state=0).fit(lle_embedding)
lle_labels = lle_kmeans.labels_
axs[1,1].scatter(lle_embedding[:,0],lle_embedding[:,1],c=lle_labels)


# spectral embedding
se = SpectralEmbedding(n_components=2)
se_embedding = se.fit_transform(pixel_data)
axs[0,2].scatter(se_embedding[:,0],se_embedding[:,1],c=high_dim_labels)
axs[0,2].set_title('spectral')
# cluster in low dim
se_kmeans = KMeans(n_clusters=3, random_state=0).fit(se_embedding)
se_labels = se_kmeans.labels_
axs[1,2].scatter(se_embedding[:,0],se_embedding[:,1],c=se_labels)

# mds
mds = MDS(n_components=2)
mds_embedding = mds.fit_transform(pixel_data)
axs[0,3].scatter(mds_embedding[:,0],mds_embedding[:,1],c=high_dim_labels)
axs[0,3].set_title('mds')
# cluster in low dim
mds_kmeans = KMeans(n_clusters=3, random_state=0).fit(mds_embedding)
mds_labels = mds_kmeans.labels_
axs[1,3].scatter(mds_embedding[:,0],mds_embedding[:,1],c=mds_labels)

# tsne
tsne = TSNE(n_components=2)
tsne_embedding = tsne.fit_transform(pixel_data)
axs[0,4].scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=high_dim_labels)
axs[0,4].set_title('tsne')
# cluster in low dim
tsne_kmeans = KMeans(n_clusters=3, random_state=0).fit(tsne_embedding)
tsne_labels = tsne_kmeans.labels_
axs[1,4].scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=tsne_labels)

# plot mode set
axs[0,0].set_ylabel('high dim clustering')
axs[1,0].set_ylabel('low dim clustering')

for ax in axs.flatten():
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(plot_path+'manifold.png')
plt.show()
