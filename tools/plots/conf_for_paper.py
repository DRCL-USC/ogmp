from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib
import numpy as np
from string import Template
import os
path2save = Template("results/plots_for_paper/$fname$version.png")

def find_latest_version(fname):
    i = 0
    while True:
        path2file = path2save.substitute(fname = fname, version = str(i))
        if not os.path.exists(path2file):
            return i
        i+=1

# print(path2save.substitute(fname = 'test_', version = str(0)))
# axis_font_size = 20
legend_font_size = 20 #25
tick_font_size = 15 #15

plt.rcParams['text.usetex'] = True
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = legend_font_size
plt.rcParams['axes.titlesize'] = legend_font_size

plt.rcParams['font.weight'] = 'bold'
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['lines.linewidth'] = 2
# plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=tick_font_size,color='black') 
plt.rc('ytick', labelsize=tick_font_size,color='black') 
matplotlib.rcParams.update({'font.size':legend_font_size})