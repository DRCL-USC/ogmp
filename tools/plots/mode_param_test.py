import sys
sys.path.append("./")
from tools.plots.conf_for_paper import *
import numpy as np
import os
# import matplotlib.pyplot as plt

import yaml
import argparse 


track_elem2mode_name = {
                        'block': 'jump',
                        'gap': 'leap',
                        'flat': 'pace'
                        }
def a4_subplot_imshow(
                        datas, 
                        layout = [1,1],
                        fig_other_axis_scale = 1,
                        vmax = None,
                        vmin = None,
                        xlabels = None,
                        ylabels = None,
                        xticks = None,
                        yticks = None,
                        titles = None,
                        squares = None,
                    ):
    # a4: align along an axis
    # note: in imshow() the first axis is y and the second is x

    if vmax is None:
        vmax = max([data.max() for data in datas])
    if vmin is None:
        vmin = 0 #min([data.min() for data in datas])

    if layout[0] == 1 and layout[1] == 1:
        height_ratios = [1]
        width_ratios = [1]

    elif layout[0] == 1:

        height_max = max([data.shape[0] for data in datas])
        datas.append(np.zeros((height_max,1)))

        # align along y (height)
        width_sum = sum([data.shape[1] for data in datas])
        width_ratios = [data.shape[1]/width_sum for data in datas]
        height_sum = 1
        height_ratios = [1]#*len(datas)
        layout[1] += 1

        print('width_ratios: ',width_ratios)
        fig_width_max = 20
        aspect_ratio = height_sum/width_sum
        fig_width  = 10.74 #fig_width_max
        fig_height =  fig_width_max*aspect_ratio*fig_other_axis_scale

    elif layout[1] == 1:

        width_max = max([data.shape[1] for data in datas])
        datas.append(np.zeros((1,width_max)))
        
        # align along x (width)
        height_sum = sum([data.shape[0] for data in datas])
        height_ratios = [data.shape[0]/height_sum for data in datas]
        width_sum = 1
        width_ratios = [1]#*len(datas)
        layout[0] += 1

        fig_height_max = 18
        aspect_ratio = width_sum/height_sum
        fig_height =  fig_height_max
        fig_width = fig_height_max*aspect_ratio*fig_other_axis_scale
    
    else:
        raise NotImplementedError
        pass

    

    # print('fig size: ',fig_width, fig_height)

    fig  = plt.figure(
                        constrained_layout=True,
                        figsize=(
                                    fig_width, 
                                    fig_height  
                                )
                        )

    specs = fig.add_gridspec(
                            nrows=layout[0], 
                            ncols=layout[1], 
                            width_ratios=width_ratios,
                            height_ratios=height_ratios,
                            
                            )
    if isinstance(specs,np.ndarray):
        if len(specs.shape) == 1:
            specs = specs
        elif len(specs.shape) == 2:
            specs = specs.flatten()

    
    
    
    
    k = 0
    idx_except_last = layout[1]-1 if layout[0]==1 else layout[0]-1

    for spec,data in zip(specs,datas):
        if k != idx_except_last:
            ax = fig.add_subplot(spec)
            im = ax.imshow(
                        data,
                        cmap='Reds', 
                        vmax = vmax,
                        vmin = vmin,
                    )
            if xticks is not None:
                ax.set_xticks(
                              ticks = range(len(xticks[k])),
                              labels = xticks[k],
                              rotation = 45

                              )
            if yticks is not None:
                ax.set_yticks(
                              ticks = range(len(yticks[k])),
                              labels = yticks[k],
                              rotation = 45
                              )
            if xlabels is not None:
                ax.set_xlabel(xlabels[k])
            if ylabels is not None:
                ax.set_ylabel(ylabels[k])  
            if titles is not None:
                ax.set_title(titles[k])
            if squares is not None:
                if squares[k] is not None:
                    ax.plot(
                            squares[k][:,0],
                            squares[k][:,1],
                            color='black',
                            linewidth=2,
                            )
                    # annotate text to the center of the square
                    ax.text(
                            np.mean(squares[k][:,0]),
                            np.mean(squares[k][:,1]),
                            r'$\Psi_{'+track_elem2mode_name[titles[k]]+r'}$',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=legend_font_size,
                            )


                              
        k+=1


    # fig.subplots_adjust(right=0.5)

    if layout[0] == 1 :
        total_width_except_last = sum(width_ratios[:-1])
        
        cbar_ax = fig.add_axes([
                                total_width_except_last, 
                                0.05, 
                                0.01, 
                                0.9
                                ])
        fig.colorbar(im, cax=cbar_ax)

    else:
        total_height_except_last = sum(height_ratios[:-1])
        cbar_ax = fig.add_axes([
                                0.05, 
                                1.0-total_height_except_last, 
                                0.94, 
                                0.01
                                ])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    plt.savefig(args.path2logs+'/plots/mpt_all_modes.png')
    plt.show()
    

param2cord = lambda p,pl,pu,len: len * (p - pl) / (pu - pl)


st_line_y = lambda x,p1,p2: (p2[1]-p1[1])/(p2[0]-p1[0])*(x-p1[0])+p1[1] 


parser = argparse.ArgumentParser()
parser.add_argument("--path2logs", default="", type=str)  

args = parser.parse_args()

os.makedirs(args.path2logs+'/plots/',exist_ok=True) 

exp_name = args.path2logs.split('/')[2]+'/'+args.path2logs.split('/')[3]
print('path2logs: ',args.path2logs)
print('exp_name: ',exp_name)

modes = [ mode_name for mode_name in os.listdir(args.path2logs) if mode_name not in ['plots']  ]

# load policy training config
policy_logpath = './logs/'+exp_name+'/'
tstng_exp_conf_file = open(policy_logpath+'/exp_conf.yaml') 
tstng_exp_conf = yaml.load(tstng_exp_conf_file, Loader=yaml.FullLoader)


# make figure
data_to_plot = []
xlabels = []
ylabels = []
xticks = []
yticks = []
titles = []
trng_dist_supports = []
for mi,mode_name in enumerate(modes):
    print(mi,'mode_name: ',mode_name)
    path2logs = args.path2logs+mode_name+'/'
    
    
    # load log files
    files = os.listdir(path2logs)
    
    # sort the files as the order of the number of the task
    files = [file for file in files if file.endswith('.npz')]
    files.sort(key=lambda x: int(x.split('.')[0]))
    returns = []
    total_steps = []
    param_vals = []
    for file_name in files:
        file_path = path2logs + file_name

        data = np.load(file_path, allow_pickle=True)
        
        returns.append(data['returns'].tolist())
        total_steps.append(data['epi_len'].tolist())
        param_vals.append(data['param_val'].tolist())
        
    
        # load policy mode param config, to get track element name
    anlys0_log_path =path2logs+'0.yaml'
    anlys0_log_conf_file = open(anlys0_log_path) # remove
    anlys0_log_conf = yaml.load(anlys0_log_conf_file, Loader=yaml.FullLoader)
    track_element = list(anlys0_log_conf['task']['track_elements'].keys())[0]
    param_names = anlys0_log_conf['task']['track_elements'][track_element]['param_names']

    # set prams of interest(poi): pioi[0] # y axis,  pioi[1] # x axis
    if 'block' in mode_name:
        track_element = 'block'
        pioi = [1,2] # v2 # length, height
        # pioi = [0,2] # v3 # start, height
        param_names[pioi[0]] = param_names[pioi[0]]+r' $(w)$'
        param_names[pioi[1]] = param_names[pioi[1]]+r' $(h)$'

    elif 'gap' in mode_name:
        track_element = 'gap'
        pioi = [0,1]

        param_names[pioi[0]] = r'approach $(w)$'
        param_names[pioi[1]] = r'distance $(d)$'

    elif 'flat' in mode_name:
        track_element = 'flat'
        param_names = ['heading speed']
        pioi = [0]

    # plot the returns grid
    if len(pioi) == 1:
        param_y_unq = np.unique(np.array(param_vals)[:,pioi[0]])
        param_x_unq = np.array([0])
    else:

        param_y_unq = np.unique(np.array(param_vals)[:,pioi[0]])
        param_x_unq = np.unique(np.array(param_vals)[:,pioi[1]])

    ncols = param_x_unq.shape[0]
    nrows = param_y_unq.shape[0]

    returns2d = np.array(returns).reshape(nrows,ncols)
    len2d = np.array(total_steps).reshape(nrows,ncols)
    
    
    if len(pioi) == 1:
        pioi.append(-1)
        param_names.append('')
        
    
    if mode_name in ['gap','block']:
        # to draw the training distribution's support
        support = tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['support']

        # bug, the support transformation is not correct
        print('support: ',support)

        top_y = st_line_y(
                        support[0][pioi[0]],
                        (param_y_unq[0],0),
                        (param_y_unq[-1],returns2d.shape[0]-1)
                        )
            

        bot_y = st_line_y(
                            support[1][pioi[0]],
                            (param_y_unq[0],0),
                            (param_y_unq[-1],returns2d.shape[0]-1)
                            )
        left_x = st_line_y(
                            support[0][pioi[1]],
                            (param_x_unq[0],0),
                            (param_x_unq[-1],returns2d.shape[1]-1)
                            )
        right_x = st_line_y(
                            support[1][pioi[1]],
                            (param_x_unq[0],0),
                            (param_x_unq[-1],returns2d.shape[1]-1)
                            )


        support_square =np.array( [
                    [left_x,top_y],
                    [left_x,bot_y],
                    [right_x,bot_y],
                    [right_x,top_y],
                    [left_x,top_y],
                ])
    else:
        support_square = None

    data_to_plot.append(returns2d)
    xlabels.append(param_names[pioi[1]])
    ylabels.append(param_names[pioi[0]])



    xticks.append(param_x_unq.round(2))
    yticks.append(param_y_unq.round(2))

    title = mode_name
    # +", "+ r'$\Psi_{'+track_elem2mode_name[track_element]+r'}$'
    titles.append(title)
    trng_dist_supports.append(support_square)


a4_subplot_imshow(
                    data_to_plot, 
                    layout = [1,len(data_to_plot)],
                    # layout = [len(data_to_plot),1],
                    fig_other_axis_scale = 5,
                    xlabels=xlabels,
                    ylabels=ylabels,
                    xticks=xticks,
                    yticks=yticks,
                    titles=titles,
                    squares = trng_dist_supports
                )


exit()


for mi,mode_name in enumerate(modes):
    print(mi,'mode_name: ',mode_name)
    path2logs = args.path2logs+mode_name+'/'
    

    # load policy mode param config, to get track element name
    anlys0_log_path =path2logs+'0.yaml'
    anlys0_log_conf_file = open(anlys0_log_path) # remove
    anlys0_log_conf = yaml.load(anlys0_log_conf_file, Loader=yaml.FullLoader)
    track_element = list(anlys0_log_conf['task']['track_elements'].keys())[0]
    param_names = anlys0_log_conf['task']['track_elements'][track_element]['param_names']


    # set prams of interest(poi): pioi[0] # y axis,  pioi[1] # x axis
    if 'block' in mode_name:
        track_element = 'block'
        pioi = [1,2]

        ax = fig.add_subplot(gs[:, 0:4])

    elif 'gap' in mode_name:
        track_element = 'gap'
        pioi = [0,1]
        ax = fig.add_subplot(gs[:, 1:4])
    elif 'flat' in mode_name:
        track_element = 'flat'
        pioi = [0]
        ax = fig.add_subplot(gs[:, -1])


    # plot the returns grid
    if len(pioi) == 1:
        param_y_unq = np.unique(np.array(param_vals)[:,pioi[0]])
        param_x_unq = np.array([0])
    else:
        param_y_unq = np.unique(np.array(param_vals)[:,pioi[0]])
        param_x_unq = np.unique(np.array(param_vals)[:,pioi[1]])



    ncols = param_x_unq.shape[0]
    nrows = param_y_unq.shape[0]

    returns2d = np.array(returns).reshape(nrows,ncols)
    len2d = np.array(total_steps).reshape(nrows,ncols)

    # may have to generalise later
    aspect_ratio = len(param_x_unq) / len(param_y_unq)
    
    '''
    if tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['type'] == 'grid':
        support = tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['support']

        top_y = support[0][pioi[0]]
        bot_y = support[1][pioi[0]]

        left_x = support[0][pioi[1]]
        right_x = support[1][pioi[1]]


        # top line
        plt.plot([left_x,right_x], [top_y,top_y], color='k', linewidth=1)
        # bottom line
        plt.plot([left_x,right_x], [bot_y,bot_y], color='k', linewidth=1)
        # left line
        plt.plot([left_x,left_x], [top_y,bot_y], color='k', linewidth=1)
        # right line
        plt.plot([right_x,right_x], [top_y,bot_y], color='k', linewidth=1)
        
        
        density_x = 3
        density_y = 3
        ys = np.linspace(support[0][pioi[0]],support[1][pioi[0]],density_y)
        xs = np.linspace(support[0][pioi[1]],support[1][pioi[1]],density_x)
        for y in ys:
            for x in xs:
                plt.scatter(x,y,marker='o',color='b',s=50)
        
        plt.ylabel(param_names[pioi[0]])
        plt.xlabel(param_names[pioi[1]])
        plt.xticks(
                    ticks=xs,
                    labels=np.array(xs).round(2),
                    rotation=0
                    )
        plt.yticks(
                    ticks=ys,
                    labels=np.array(ys).round(2),
                    rotation=0
                    )
        plt.title('params range of mode:'+mode_name)
        plt.tight_layout()
        plt.grid()
        plt.savefig(args.path2logs+'/plots/'+mode_name+'_param_set.png')
        plt.show()
        plt.close('all')


        if mode_name == 'block':
            support_a = 0.25
            support_b = 1.5
            plt.hlines(
                        0,
                        support_a,
                        support_b,
                        colors='k',
                        linewidth=1
                        )
            
            plt.scatter(
                        np.linspace(support_a,support_b,9),
                        np.zeros(9),
                        marker='o',
                        color='b',
                        s=50
                        )
            plt.yticks([])
            plt.xticks(
                        ticks=np.linspace(support_a,support_b,9),
                        labels=np.linspace(support_a,support_b,9).round(2),
                        rotation=0
                        )
            plt.xlabel('goal')
            plt.title('params range of mode: flat')
            plt.tight_layout()
            plt.grid()
            plt.savefig(args.path2logs+'/plots/flat_param_set.png')
            plt.show()
            plt.close('all')
    '''


    ax.set_title(mode_name)
    im = ax.imshow( 
                returns2d, 
                cmap='Reds', 
                interpolation='nearest',
                vmin=min(0,min(returns)),
                vmax=np.max(returns),
            )
    
    # divider = make_axes_locatable(axs[0])
    if len(pioi) == 1:
        # cax = divider.append_axes('right', size='50%', pad=0.1)
        pioi.append(-1)
        param_names.append('')

    # else:
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical') 
    
    # axs[1].set_title('episode length')
    # im = axs[1].imshow( 
    #             len2d, 
    #             cmap='seismic', 
    #             interpolation='nearest',
    #             vmin=min(0,min(total_steps)),
    #             vmax=max(total_steps),
    #         )
    # divider = make_axes_locatable(axs[1])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical') 

        
    ax.set_ylabel(param_names[pioi[0]])
    ax.set_yticks( 
                ticks=range(len(param_y_unq)),
                labels=np.array(param_y_unq).round(2),
                rotation=0
                )
    
    ax.set_xlabel(param_names[pioi[1]])
    ax.set_xticks(
                ticks=range(len(param_x_unq)),
                labels=np.array(param_x_unq).round(2),
                rotation=90
                )


    # plot the training distribution's support
    if track_element in tstng_exp_conf['task']['track_elements']:
        if tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['type'] == 'discrete':

            terrain_param = tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['points']

            for trng_prm in terrain_param:

                y = returns2d.shape[0] * (trng_prm[pioi[0]] - param_y_unq[0]) / (param_y_unq[-1] - param_y_unq[0]) 
                x = returns2d.shape[1] * (trng_prm[pioi[1]] - param_x_unq[0]) / (param_x_unq[-1] - param_x_unq[0])
                ax.scatter(x,y,marker='x',color='k',s=10)

        elif tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['type'] == 'discrete_grid':
            
            
            support = tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['support']
            density = tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['density']


            for py in np.linspace(support[0][pioi[0]],support[1][pioi[0]],density[pioi[0]]):
                for px in np.linspace(support[0][pioi[1]],support[1][pioi[1]],density[pioi[1]]):
            
                    
                    y = param2cord(py,param_y_unq[0],param_y_unq[-1],returns2d.shape[0])
                    x = param2cord(px,param_x_unq[0],param_x_unq[-1],returns2d.shape[1])
                    ax.scatter(x,y,marker='x',color='k',s=10)

        elif tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['type'] == 'grid':
            support = tstng_exp_conf['task']['track_elements'][track_element]['param_dist']['support']

            top_y = param2cord(support[0][pioi[0]],param_y_unq[0],param_y_unq[-1],returns2d.shape[0])
            bot_y = param2cord(support[1][pioi[0]],param_y_unq[0],param_y_unq[-1],returns2d.shape[0])

            left_x = param2cord(support[0][pioi[1]],param_x_unq[0],param_x_unq[-1],returns2d.shape[1])
            right_x = param2cord(support[1][pioi[1]],param_x_unq[0],param_x_unq[-1],returns2d.shape[1])

            # top line
            ax.plot([left_x,right_x], [top_y,top_y], color='k', linewidth=1)
            # bottom line
            ax.plot([left_x,right_x], [bot_y,bot_y], color='k', linewidth=1)
            # left line
            ax.plot([left_x,left_x], [top_y,bot_y], color='k', linewidth=1)
            # right line
            ax.plot([right_x,right_x], [top_y,bot_y], color='k', linewidth=1)


# divider = make_axes_locatable(axs[-1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im, cax=cax, orientation='vertical') 
# save the plot      
# plt.tight_layout()
plt.savefig(args.path2logs+'/plots/mpt_all_modes.png')


plt.show()
plt.close('all')


