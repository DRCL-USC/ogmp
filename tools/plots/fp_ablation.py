import sys
sys.path.append("./")
from tools.plots.conf_for_paper import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

min_alpha = 0.3
# load data 
print('group 1')
data4grp1 = np.load('./results/fgb_vary_rho/metrics.npz')
labels4grp1 = np.loadtxt('results/fgb_vary_rho/labels.txt',dtype=str)

for metric in data4grp1.files:
    for label in labels4grp1:
        print(label, data4grp1[metric][labels4grp1==label].shape)
print('group 2')
data4grp2 = np.load('./results/fgb_vary_horizon/metrics.npz')
labels4grp2 = np.loadtxt('results/fgb_vary_horizon/labels.txt',dtype=str)

for metric in data4grp2.files:
    for label in labels4grp2:
        print(label, data4grp2[metric][labels4grp2==label].shape)
print('group 3')
data4grp3 = np.load('./results/fgb_vary_obs/metrics.npz')
labels4grp3 = np.loadtxt('results/fgb_vary_obs/labels.txt',dtype=str)

for metric in data4grp3.files:
    for label in labels4grp3:
        print(label, data4grp3[metric][labels4grp3==label].shape)
print('group 4')
data4grp4 = np.load('./results/dive_rp_vary_oracle/metrics.npz')
labels4grp4 = np.loadtxt('results/dive_rp_vary_oracle/labels.txt',dtype=str)

for metric in data4grp4.files:
    for label in labels4grp4:
        print(label, data4grp4[metric][labels4grp4==label].shape)


# make figure
fig0 = plt.figure(figsize=(20,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], width_ratios=[3,3,8,8,])
gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[6,6,15])

# add dummpy subplots
axs = []
for i in range(4):
    axs.append(fig0.add_subplot(gs1[i]))
for i in range(3):
    axs.append(fig0.add_subplot(gs2[i]))

# plot group 1
for i in [2,3]:
    metric = data4grp1.files[i-2]
    data = data4grp1[metric]


    # alpha increases from min_alpha to 1 

    for j,data_per_bar in enumerate(data):
        alpha = (1 - min_alpha)*(j)/(len(data)-1) + min_alpha
        axs[i].bar(
                    j,
                    data_per_bar,
                    # yerr=np.std(data),
                    # label=
                    color='C0',
                    alpha=alpha
                    )

    # put a vertical and horizontal line at the max value
    max_id = np.argmax(data)
    axs[i].axvline(x=max_id, color='k', linestyle='--', linewidth=1)
    axs[i].axhline(y=data[max_id], color='k', linestyle='--', linewidth=1)

    if i == 3:
        # put a vline at max possible value
        axs[i].axhline(y=10.0, color='r', linestyle='--', linewidth=1)
        axs[i].set_ylabel(r"$J_\mathcal{T}$")
    else:
        axs[i].set_ylabel(r"$\tilde{J}_\mathcal{T}$")
    # set ylabel

    # set xticks
    # labels_filtered = [ label.replace('prev_orac_',' p=') for label in ]
    labels_filtered = []
    for label in labels4grp1:
        labels_filtered.append(label.replace('prev_orac_',r"$\rho$="))
    axs[i].set_xticks(
                        np.arange(len(labels_filtered)), 
                        labels_filtered,
                        rotation=0
                    )
    
# plot group 2
for i in [0,1]:
    metric = data4grp2.files[i]
    data = data4grp2[metric]
    for j,data_per_bar in enumerate(data):
        alpha = (1 - min_alpha)*(j)/(len(data)-1) + min_alpha
        axs[i].bar(
                    j,
                    data_per_bar,
                    # yerr=np.std(data),
                    # label= 
                    color='C1',
                    alpha=alpha,
                    )
    # put a vertical and horizontal line at the max value
    max_id = np.argmax(data)
    axs[i].axvline(x=max_id, color='k', linestyle='--', linewidth=1)
    axs[i].axhline(y=data[max_id], color='k', linestyle='--', linewidth=1)

    if i == 1:
        # put a vline at max possible value
        axs[i].axhline(y=10.0, color='r', linestyle='--', linewidth=1)
        axs[i].set_ylabel(r"$J_\mathcal{T}$")
    else:
        axs[i].set_ylabel(r"$\tilde{J}_\mathcal{T}$")
    # set xticks
    labels_filtered = [ label.replace('H',r"$\frac{\Delta t}{dt}$") for label in labels4grp2]
    axs[i].set_xticks(
                        np.arange(len(labels_filtered)), 
                        labels_filtered,
                        rotation=0
                    )

# plot group 3
for i in [4,5]:
    metric = data4grp3.files[i-4]
    data = data4grp3[metric]
    for j,data_per_bar in enumerate(data):
        alpha = (1 - min_alpha)*(j)/(len(data)-1) + min_alpha
        axs[i].bar(
                    j,
                    data_per_bar,
                    # yerr=np.std(data),
                    # label= 
                    color='C2',
                    alpha=alpha,
                    )
    # put a vertical and horizontal line at the max value
    max_id = np.argmax(data)
    axs[i].axvline(x=max_id, color='k', linestyle='--', linewidth=1)
    axs[i].axhline(y=data[max_id], color='k', linestyle='--', linewidth=1)

    if i == 5:
        # put a vline at max possible value
        axs[i].axhline(y=10.0, color='r', linestyle='--', linewidth=1)
        axs[i].set_ylabel(r"$J_\mathcal{T}$")
    else:
        axs[i].set_ylabel(r"$\tilde{J}_\mathcal{T}$")


    label_transform = {
                        '[z]': "["+r"$z_t$"+"]",
                        '[z,c]': "["+r"$z_t$"+", "+r"$c_t$"+"]",
                        '[h]': "["+r"$h_t$"+"]",
                        '[h,c]': "["+r"$h_t$"+", "+r"$c_t$"+"]",
                        '[z,h]': "["+r"$z_t$"+", "+r"$h_t$"+"]",
                        '[z,h,c]': "["+r"$z_t$"+", "+r"$h_t$"+", "+r"$c_t$"+"]",
    }

    labels_filtered = [label_transform[label] for label in labels4grp3]

    
        
    # set xticks
    axs[i].set_xticks(
                        np.arange(len(labels_filtered)), 
                        labels_filtered,
                        rotation=90
                    )

# plot group 4
for i in [6]:
    metric = data4grp4.files[0]
    data = data4grp4[metric]
    
    
    
    labels_filtered = [ label.replace('_orac_','\n') for label in labels4grp4]
    


    for j,data_per_bar in enumerate(data):
        if 'li' in labels_filtered[j]:
            color = 'C4'
        elif 'lqr' in labels_filtered[j]:
            color = 'C5'
        else:
            color = 'C8'

        alpha = (1 - min_alpha)*(j)/(len(data)-1) + min_alpha
        axs[i].bar(
                    j,
                    data_per_bar,
                    # yerr=np.std(data),
                    # label= 
                    color=color,
                    alpha=alpha,
                    )
    # put a vertical and horizontal line at the max value
    max_id = np.argmax(data)
    axs[i].axvline(x=max_id, color='k', linestyle='--', linewidth=1)
    axs[i].axhline(y=data[max_id], color='k', linestyle='--', linewidth=1)

    axs[i].set_ylabel(r"$\tilde{J}_\mathcal{T}$")
    # set xticks
    axs[i].set_xticks(
                        np.arange(len(labels_filtered)), 
                        labels_filtered,
                        rotation=0
                    )
    # draw a line connecting bars with index in a_idx_list 
    a_idx_list = [3,8,10]

    axs[i].plot(
                a_idx_list,
                data[a_idx_list],
                color='C0',
                linestyle='-',
                linewidth=1.5
                )
    axs[i].scatter(
                    a_idx_list,
                    data[a_idx_list],
                    color='C0',
                    marker='s',
                    s=20
                    )
    # annotate text "alpha" at the mid point os a_idx_list[0] and a_idx_list[1]
    anotate_at_x = (a_idx_list[0] + a_idx_list[1])/2
    anotate_at_y = (data[a_idx_list[0]] + data[a_idx_list[1]])/2
    axs[i].annotate(
                    r"$\alpha(\epsilon)$",
                    xy=(anotate_at_x, anotate_at_y),
                    xytext = (anotate_at_x-2, anotate_at_y+10.0),
                    arrowprops=dict(
                                    facecolor='black',
                                    edgecolor='black',
                                    arrowstyle='->',
                                    ),
                    )
        

fig0.tight_layout()
# set fig wspace to 0.23
fig0.subplots_adjust(wspace=0.23)

# draw bounding boxes

# # group plots 0, 1 and add a common bounding box around them 
# bbox0 = axs[0].get_window_extent()
# bbox1 = axs[1].get_window_extent()

# total_width = bbox1.x1 - bbox0.x0 - 55
# total_height = bbox1.y1 - bbox0.y0

# pad = 0.0
# rect0 = plt.Rectangle(
#                         (bbox0.x0- pad*bbox0.width, bbox0.y0- pad*bbox0.height), 
#                         total_width,
#                         total_height,
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect0)

# # group plots 2, 3 and add a common bounding box around them 
# bbox2 = axs[2].get_window_extent()
# bbox3 = axs[3].get_window_extent()

# total_width = bbox3.x1 - bbox2.x0 
# total_height = bbox3.y1 - bbox2.y0

# pad = 0.0
# rect1 = plt.Rectangle(
#                         (bbox2.x0- pad*bbox2.width, bbox2.y0- pad*bbox2.height), 
#                         total_width,
#                         total_height,
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect1)

# # group plots 4, 5, and add a common bounding box around them
# bbox0 = axs[4].get_window_extent()
# bbox1 = axs[5].get_window_extent()

# total_width = bbox1.x1 - bbox0.x0 
# total_height = bbox1.y1 - bbox0.y0

# pad = 0.0
# rect2 = plt.Rectangle(
#                         (bbox0.x0- pad*bbox0.width, bbox0.y0- pad*bbox0.height), 
#                         total_width,
#                         total_height,
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect2)

# # group plot6 and add a bounding box around it
# bbox0 = axs[6].get_window_extent()

# total_width = bbox0.x1 - bbox0.x0
# total_height = bbox0.y1 - bbox0.y0

# pad = 0.0
# rect3 = plt.Rectangle(
#                         (bbox0.x0- pad*bbox0.width, bbox0.y0- pad*bbox0.height), 
#                         total_width,
#                         total_height,
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect3)





# get bounding box of the first subplot
# bbox0 = axs[4].get_window_extent()
# print(bbox0.x0, bbox0.y0, bbox0.x1, bbox0.y1)
# print(bbox0.width, bbox0.height)
# print(bbox0.x1- bbox0.x0, bbox0.y1- bbox0.y0)

# # draw the bounding box
# pad = 0.0
# rect0 = plt.Rectangle(  
#                         (bbox0.x0- pad*bbox0.width, bbox0.y0- pad*bbox0.height), 
#                         bbox0.width + 2*pad*bbox0.width,
#                         bbox0.height + 2*pad*bbox0.height, 
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect0)




# bbox1 = axs[1].get_window_extent()

# # fig0.patches.append(rect0)
# rect1 = plt.Rectangle(
#                         (bbox0.x1- pad*bbox0.width, bbox0.y1- pad*bbox0.height), 
#                         total_width,
#                         total_height,
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect1)
# rect0 = plt.Rectangle(
#                         (bbox0.x0- pad*bbox0.width, bbox0.y0- pad*bbox0.height), 
#                         bbox0.width + 2*pad*bbox0.width,
#                         bbox0.height + 2*pad*bbox0.height, 
#                         fill=False, 
#                         color='blue'
#                         )
# fig0.patches.append(rect0)

plt.show()




exit()



# make fig1 with 4 subplots in a row, with the same height ratio and width ratios : 8:8:3:3
fig1 = plt.figure(figsize=(20,3)) 
gs = fig1.add_gridspec(1, 4, width_ratios=[8,8,3,3])
# add dummpy subplots
axs = []
for i in range(4):
    axs.append(fig1.add_subplot(gs[i]))
    axs[-1].set_facecolor("red")
    axs[-1].label_outer()
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])
fig1.tight_layout()

# make fig2 with 3 subplots in a row, with the same height ratio and width ratios : 6:6:15
fig2 = plt.figure(figsize=(20,3))
gs = fig2.add_gridspec(1, 3, width_ratios=[6,6,15])
# add dummpy subplots
axs = []
for i in range(3):
    axs.append(fig2.add_subplot(gs[i]))
    axs[-1].set_facecolor("red")
    axs[-1].label_outer()
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])
fig2.tight_layout()
