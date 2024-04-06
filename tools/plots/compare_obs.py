# compares vary_oracle and vary_rho experiments 
import sys
import os
import csv
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt

def read_all_logs(path2logs):
    loglist = os.listdir(path2logs)
    loglist = [file for file in loglist if '.npz' in file]
    metrics_dict = {'displacement':[]}
    mode_latents = None
    # collect all metrics
    for i,log_name in enumerate(loglist):
        worker_log = np.load(os.path.join(path2logs,log_name),allow_pickle=True)
        
        if mode_latents is None:
            mode_latents = worker_log['mode_latents_encountered']
        else:
            mode_latents = np.append(mode_latents,worker_log['mode_latents_encountered'],axis=0)
        base_poss_f = worker_log['base_poss_f']
        base_poss_i = worker_log['base_poss_i']
        displacement = np.linalg.norm(base_poss_f- base_poss_i,axis=1)
        metrics_dict['displacement'] += displacement.tolist()
        for file in worker_log.files:
            if file in metrics_to_plot:

                if file not in metrics_dict.keys():
                    metrics_dict[file] = []

                metrics_dict[file] += worker_log[file].tolist()
    
    # do only if mode_latents is 2d
    if len(mode_latents.shape) == 2:
        print('mode_latents',mode_latents.shape)
        # get the centre of mode latents
        a_ID = np.mean(mode_latents,axis=0)
        # get eh max distance from centre
        d_ID = np.max(np.linalg.norm(mode_latents-a_ID,axis=1))
        print('d_ID',d_ID, 'a_ID',a_ID)
        # get the max distance


    return metrics_dict



# fgb_vary_obs
path_substr1 = './results/fgb_vary_obs/'
path_substr2 = '/n_rollouts_test/100_rollouts_10m_track_sc'
n_variants = 6
params_vary_path = './logs/fgb_vary_obs/param_vary_list.csv'

paths2logs = [path_substr1 + str(i) + path_substr2 for i in range(n_variants)]


# read csv from params_vary_path
plot_order = [
                1,0,
                3,2,
                5,4
            ]
obs_label = [
                '[z]',
                '[z,c]',
                '[h]',
                '[h,c]',
                '[z,h]',
                '[z,h,c]'
            ]

metrics_to_plot = [
                    # 'episode_length',
                    'undisc_returns',
                    'displacement',

                ]
max_id_each_metric = {}
n_metrics = None
k = 0
ordered_data = {}

for i,key in enumerate(metrics_to_plot):
    ordered_data[key] = []
for pi,ol in zip(plot_order,obs_label):
    
    path2logs = paths2logs[pi]
    metrics_dict = None
    
    # try:
    metrics_dict = read_all_logs(path2logs)
    # except:
    #     print('skipping',path2logs)
    #     continue

    if metrics_dict is not None:
        # if first iteration, initialize
        if n_metrics is None:
            n_metrics = len(metrics_dict)
            
            fig, axs = plt.subplots(1,n_metrics,figsize=(20,10))#int(20/n_metrics)))

        
        # bar plot, comparing different logs
        color = None
        if 'z,h' in ol:
            color = 'red'
        elif 'h' in ol:
            color = 'green'
        elif 'z' in ol:
            color = 'blue'
        alpha = 1.0 if 'c' in ol else 0.5
        print(path2logs, color, alpha)
        
        for i,key in enumerate(metrics_dict.keys()):

            data = np.mean(metrics_dict[key])
            title = key

        
            axs[i].bar(
                        k,
                        data,
                        yerr=np.std(data),
                        # label= 
                        color=color,
                        alpha=alpha,
                        )
            axs[i].set_title(title)


            axs[i].set_xlabel('oracles')

            # check if this is max and update
            if key not in max_id_each_metric.keys():
                max_id_each_metric[key] = [k,data]
            else:
                if data > max_id_each_metric[key][1]:
                    max_id_each_metric[key] = [k,data]
            ordered_data[key].append(data)
        
        k+=1

# plot max values with horizontal and vertical lines
for i,key in enumerate(metrics_dict.keys()):
    axs[i].axvline(x=max_id_each_metric[key][0], color='r', linestyle='--')
    axs[i].axhline(y=max_id_each_metric[key][1], color='r', linestyle='--')
    axs[i].set_xticks(
                        np.arange(len(obs_label)), 
                        obs_label,
                        rotation=0
                    )
    axs[i].grid()
    # axs[i].set_ylabel(key)
    # axs[i].set_xlabel('oracles')
    # axs[i].legend()


fig.suptitle('metrics of varied observations')
fig.tight_layout()
plt.savefig(path_substr1+'/metrics.png')
np.savez_compressed(path_substr1+'/metrics',**ordered_data)
np.savetxt(path_substr1+'/labels.txt',obs_label,fmt='%s')
plt.show()