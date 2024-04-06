import os
import subprocess


path2folder = '/home/loki/Downloads/drive-download-20240403T234139Z-001/' #'./results/each_trans/mped_wl2n/0'

for file in os.listdir(path2folder):
    
    if '.mp4' in file:
        print(file)
        in_file = os.path.join(path2folder, file)+' '
        out_file = os.path.join(path2folder, file.replace('.mp4','.gif'))+' '
        cmnd_str1 = "ffmpeg -i " 
        cmnd_str2 = "-vf \"fps=30,scale=600:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:reserve_transparent=0[p];[s1][p]paletteuse\" -y "
        os.system(cmnd_str1+in_file+cmnd_str2+out_file)
# ffmpeg -i tap_dancing.gif -ss 0 -to 1.5 -c copy output.gif