import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# make figure with 4 subfigures
# in subfigure1 : 2 plots with same height ratio, width ratio 8:8 with background color red
# in subfigure2: 2 plots with same height ratio, width ratio 3:3 with background color blue
# in subfigure3: 2 plots with same height ratio, width ratio 6:6 with background color green
# in subfigure4: 1 plot with width ratio 15 with background color yellow

fig = plt.figure(figsize=(20,3))
gs = fig.add_gridspec(2, 4, width_ratios=[8,8,3,3])


subfig1 = fig.add_subplot(gs[0,0:2])
subfig1.set_facecolor("red")
subfig1 = fig.add_subplot(gs[0,2:4])
subfig1.set_facecolor("blue")

# add dummpy subplots
axs = []
for i in range(4):
    axs.append(fig.add_subplot(gs[0,i]))
    if i in [0,1]:
        axs[-1].imshow(np.random.rand(1,8))
    else:
        axs[-1].imshow(np.random.rand(1,3))
    # axs[-1].set_facecolor("red")
    # axs[-1].label_outer()
    # axs[-1].set_xticks([])
    # axs[-1].set_yticks([])
# fig.tight_layout()

# subfig1 = fig.add_subplot(gs[0,0:2])
# subfig1.set_facecolor("red")
# subfig1 = fig.add_subplot(gs[0,2:4])
# subfig1.set_facecolor("blue")

plt.show()



