
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--path2logs", default="", type=str)  
parser.add_argument("--path2modespace", default="./logs/encoders/fgb_fx0_sρ_ae32/prtm3_vx1/dim_2/mode_space.npz", type=str)
args = parser.parse_args()

# load mode space
data = np.load(args.path2modespace)
k = 0

max_x = -np.inf
min_x = np.inf
max_y = -np.inf
min_y = np.inf

# fig = plt.figure()
for file in data.files:
    if file in ['flat','gap','block']:

        # plt.scatter(
        #                 data[file][:,0],
        #                 data[file][:,1],
        #                 label=file,
        #                 c='C'+str(k),
        #             )
        max_x = max(max_x,np.max(data[file][:,0]))
        min_x = min(min_x,np.min(data[file][:,0]))
        max_y = max(max_y,np.max(data[file][:,1]))
        min_y = min(min_y,np.min(data[file][:,1]))
        k += 1


path2logs = args.path2logs

def read_all_logs(path2logs):
    loglist = os.listdir(path2logs)
    loglist = [file for file in loglist if '.npz' in file]
    metrics_dict = {}
    
    latent_modes_encountered = []
    # collect all metrics
    for i,log_name in enumerate(loglist):
        worker_log = np.load(os.path.join(path2logs,log_name),allow_pickle=True)
        

        # if i == 0:
        #     mode_latents_encountered = worker_log['mode_latents_encountered']
        # else:
        #     mode_latents_encountered = np.vstack((mode_latents_encountered,worker_log['mode_latents_encountered']))

        for file in worker_log.files:
            
            if file not in [
                            'mode_latents_encountered',
                            'base_poss_i',
                            'base_poss_f',
                            'base_rpys_i',
                            'base_rpys_f',
                            'base_tvel_maxs',
                            'base_avel_maxs',
                            'base_tacc_maxs',
                            'base_aacc_maxs',
                            'base_tvel_means',
                            'base_avel_means',
                            'base_tacc_means',
                            'base_aacc_means',
                            
                            ]:
                if file not in metrics_dict.keys():
                    metrics_dict[file] = []
                metrics_dict[file] += worker_log[file].tolist()
            else:
                latent_modes_encountered += worker_log[file].tolist()

    return metrics_dict, latent_modes_encountered


metrics_dict, lme = read_all_logs(path2logs)



print('mode_latents_encountered:',np.array(lme).shape)
print("metrics")
for key in metrics_dict.keys():
    print('\t',key,len(metrics_dict[key]))
# mode_latents_encountered = np.unique(mode_latents_encountered,axis=0)
# rand_idx = np.random.randint(0,mode_latents_encountered.shape[0],size=1000)

# max_x = max(max_x,np.max(mode_latents_encountered[:,0]))
# min_x = min(min_x,np.min(mode_latents_encountered[:,0]))
# max_y = max(max_y,np.max(mode_latents_encountered[:,1]))
# min_y = min(min_y,np.min(mode_latents_encountered[:,1]))
# eps = 0.05
# plt.xlim(min_x-eps,max_x+eps)
# plt.ylim(min_y-eps,max_y+eps)
# plt.grid()
# scat = plt.scatter(
#             mode_latents_encountered[rand_idx[0],0],
#             mode_latents_encountered[rand_idx[0],1],
#             c='k',
#             marker='x',
#             alpha=0.5,
#             label='in trng',
#             s=50,
#             )
# def animate(i):
#     print(i)
#     scat.set_offsets(
#                         mode_latents_encountered[
#                                                 rand_idx[0:i],
#                                                 :]
#                     )
#     return [scat]



# anim = ani.FuncAnimation(fig, animate, frames=500, interval=1, blit=True,repeat=False)
# # plt.show()
# anim.save(args.path2logs+'/modes_encountered_500.gif', writer="pillow", fps=30)


# for i in tqdm(rand_idx):
#     plt.scatter(
#                 mode_latents_encountered[i,0],
#                 mode_latents_encountered[i,1],
#                 c='k',
#                 marker='x',
#                 alpha=0.3,
#                 label='in trng',
#                 s=50,
#             )
#     plt.pause(0.0001)


# plt.scatter(
#                 mode_latents_encountered[:,0],
#                 mode_latents_encountered[:,1],
#                 c='k',
#                 marker='x',
#                 alpha=0.5,
#                 label='in trng',
#             )
# plt.grid()
# plt.legend(loc='upper right')
# plt.savefig(args.path2logs+'modes_encountered.png')

# plot if required
fig2, axs = plt.subplots(2, len(metrics_dict.keys()), figsize=(20, 5))

# if len(metrics_dict.keys()) == 1:
#     axs = [axs]

fig2.suptitle('metrics')
axs[0,0].set_ylabel('metric value sample mean')
axs[0,0].set_ylabel('frequency of metric value')

k = 0
for metric_name, metric_vals in metrics_dict.items():

    print(metric_name,len(metric_vals))
    metric_mu_estimates = []
    metric_sd_estimates = []
    for i,sm in enumerate(metric_vals):
        metric_mu_estimates.append(np.mean(metric_vals[:i+1]))
        metric_sd_estimates.append(np.std(metric_vals[:i+1]))
    
    axs[0,k].plot(metric_mu_estimates)
    # plot histogram
    
    
    axs[1,k].hist(metric_vals,bins=100)
    # find mode of the histogram
    hist, bin_edges = np.histogram(metric_vals,bins=100)
    max_bin = np.argmax(hist)
    axs[1,k].axvline(bin_edges[max_bin],color='r',linestyle='--')
    

    axs[0,k].set_title(metric_name+'\n mean :'+str(round(metric_mu_estimates[-1],2)))
    axs[1,k].set_title('mode :'+str(round(bin_edges[max_bin],2)))
    axs[0,k].grid()
    axs[1,k].grid()
    axs[0,k].set_xlabel('# episodes')
    axs[1,k].set_xlabel('metric value')
    if metric_name == 'base_tvel_x_maxs':
        axs2 = axs[0,k].twinx()
        leg_length = 0.459
        fraude = np.power(metric_vals,2) / (9.81*leg_length)
        


        metric_mu_estimates = []
        for i,sm in enumerate(fraude):
            metric_mu_estimates.append(np.mean(fraude[:i+1]))

        axs2.plot(metric_mu_estimates,'r--',alpha=0.5)
        axs2.set_ylabel('fraude number '+str(round(metric_mu_estimates[-1],2)) )
    k += 1





plt.tight_layout()
plt.savefig(args.path2logs+'esm_success_rate2.png')
plt.show()
