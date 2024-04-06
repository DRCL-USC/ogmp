import sys
sys.path.append("./")
from plotters.conf_for_paper import *

import numpy as np
import argparse 

# def ccworder(A):
#     A= A - np.mean(A, 1)[:, None]
#     return np.argsort(np.arctan2(A[1, :], A[0, :]))
def downsample(data, ds_rate=10):
    return data[::ds_rate,:]

parser = argparse.ArgumentParser()
parser.add_argument("--path2modespace", default="dtsd/analysis_results/fgb_fx0_sρ_ae32/prtm3_vx1/dim_2/mode_space.npz", type=str)
args = parser.parse_args()

data = np.load(args.path2modespace)
mode_space_encoder = []
mode_space_encoder_label = []
mode_space_encoder_mean = []

fig, axs = plt.subplots(1,2)
for file in data.files:
    if file in ['flat','gap','block']:
        mode_space_encoder.append(data[file])
        mode_space_encoder_mean.append(np.mean(mode_space_encoder[-1],axis=0))
        mode_space_encoder_label.append(file)

for mode_cluster in mode_space_encoder:
    axs[0].scatter(mode_cluster[:,0],mode_cluster[:,1],s=1)
    axs[1].scatter(mode_cluster[:,0],mode_cluster[:,1],s=1)

for i,ax in enumerate(axs):
    ax.set_xlabel('latent 1')
    if i == 0:
        ax.set_ylabel('latent 2')

    # ax.set_aspect('equal', 'box')
    ax.grid()
plt.tight_layout()
plt.show()