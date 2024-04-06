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


    if 'ang_displacment' in metrics_to_plot:
        metrics_dict = {'ang_displacment':[]}
    elif 'displacement' in metrics_to_plot:
        metrics_dict = {'displacement':[]}
    else:
        metrics_dict = {}
    # collect all metrics
    for i,log_name in enumerate(loglist):
        worker_log = np.load(os.path.join(path2logs,log_name),allow_pickle=True)
        
        if 'ang_displacment' in metrics_to_plot:
            base_rpy_f = worker_log['base_rpys_f']
            base_rpy_i = worker_log['base_rpys_i']
            ang_displacment = np.linalg.norm(base_rpy_f- base_rpy_i,axis=1)
            metrics_dict['ang_displacment'] += ang_displacment.tolist()
        
        elif 'displacement' in metrics_to_plot:
            base_poss_f = worker_log['base_poss_f']
            base_poss_i = worker_log['base_poss_i']
            displacement = np.linalg.norm(base_poss_f- base_poss_i,axis=1)
            metrics_dict['displacement'] += displacement.tolist()
        
        for file in worker_log.files:
            if file in metrics_to_plot:
                if file not in metrics_dict.keys():
                    metrics_dict[file] = []

                metrics_dict[file] += worker_log[file].tolist()
    
    return metrics_dict


# fgb_policies
# path_substr1 = './results/fgb_vary_oracle/'
# path_substr2 = '/n_rollouts_test/100_rollouts_10m_track_sc'
# n_variants = 12
# params_vary_path = './logs/fgb_vary_oracle/param_vary_list.csv'
# rho_key = 'terminations/base_pos_x_ref_error_thresh'
# oracle_key = 'oracle/entry'

# fgb_vary_rho
# path_substr1 = './results/fgb_vary_rho/'
# path_substr2 = '/n_rollouts_test/100_rollouts_10m_track_sc'
# n_variants = 8
# params_vary_path = './logs/fgb_vary_rho/param_vary_list.csv'
# rho_key = 'terminations/base_pos_x_ref_error_thresh'
# oracle_key = 'prev_orac'


# dive policies
path_substr1 = './results/dive_rp_vary_oracle/'
path_substr2 = '/n_rollouts_test/100_rollouts_el120_sc'
n_variants = 20
params_vary_path = './logs/dive_rp_vary_oracle/param_vary_list.csv'
rho_key = 'terminations/base_pos_z_ref_error_thresh'
oracle_key = 'oracle/entry'



metrics_to_plot = [
                    # 'episode_length',
                    'undisc_returns',
                    # 'displacement',
                    # 'ang_displacment',

                ]


paths2logs = [path_substr1 + str(i) + path_substr2 for i in range(n_variants)]




# read csv from params_vary_path
corres_params_strs = []
rho_vals = []
oracle_names = []
avoid_names_with = [
                    'prev_ad',
                    ]
ids_2_skip = []
with open(params_vary_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for i,row in enumerate(reader):
        # print(row.keys())
        # print(row['exp_names'])
        try:        
            oracle_name = row[oracle_key].split('.')[1]
        except:
            oracle_name = oracle_key
        
        

        rho_val = float(row[rho_key])
        corres_params_strs.append(oracle_name + '_' + str(rho_val))
        rho_vals.append(rho_val)
        oracle_names.append(oracle_name)
        if oracle_name in avoid_names_with:
            ids_2_skip.append(i)
            # print('skipping',oracle_name, paths2logs[i])

arranged_ids = np.argsort(corres_params_strs).tolist()






# remove ids_2_skip


# unique values
n_rhos = np.unique(rho_vals).shape[0]
uniq_oracles = np.unique(oracle_names)
rho_max = np.max(rho_vals)
rho_min = np.min(rho_vals)


max_id_each_metric = {}
n_metrics = None

ordered_data = {}

for i,key in enumerate(metrics_to_plot):
    ordered_data[key] = []

arranged_labels = [] #np.array(corres_params_strs)[arranged_ids]
k = 0
for pi in arranged_ids:
    
    if pi not in ids_2_skip:
        
        arranged_labels.append(corres_params_strs[pi])
        path2logs = paths2logs[pi]
        metrics_dict = None
        try:
            metrics_dict = read_all_logs(path2logs)
        except:
            print('skipping',path2logs)
            continue

        if metrics_dict is not None:
            # if first iteration, initialize
            if n_metrics is None:
                n_metrics = len(metrics_dict)
                fig, axs = plt.subplots(1,n_metrics,figsize=(20,10))#int(20/n_metrics)))
                if n_metrics == 1:
                    axs = [axs]
                print('n_metrics',n_metrics)
            
            # bar plot, comparing different logs
            color = None
            oracle_name = oracle_names[pi]
            for i, uni_oracle_name in enumerate(uniq_oracles):
                if uni_oracle_name in oracle_name:
                    color = 'C' + str(i)
                    break
            rho = rho_vals[pi]
            # alpha = 1/n_rhos at rho= rho_min,  alpha=1 at rho=rho_max
            alpha = (1 - 1/n_rhos)*(rho - rho_min)/(rho_max - rho_min) + 1/n_rhos
            
            print(path2logs, color, alpha)
            
            for i,key in enumerate(metrics_dict.keys()):

                data = np.mean(metrics_dict[key])
                title = key
                # if key == 'norm_x_dist':
                #     title = 'x_dist'
                #     data = 10.0*data
                # elif key == 'norm_t_alive':
                #     title = 't_alive'
                #     data = 400*data
            
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
                        np.arange(len(arranged_labels)), 
                        arranged_labels,
                        rotation=90
                    )
    axs[i].grid()
    # axs[i].set_ylabel(key)
    # axs[i].set_xlabel('oracles')
    # axs[i].legend()


fig.suptitle('metrics of varied oracles across rho values')
fig.tight_layout()
plt.savefig(path_substr1+'/metrics.png')
np.savez_compressed(path_substr1+'/metrics',**ordered_data)
np.savetxt(path_substr1+'/labels.txt',arranged_labels,fmt='%s')
plt.show()