# for recording episodes
from PIL import Image
import mediapy as media
import glfw
import os
from dm_control import mujoco
import numpy as np
from .misc_funcs import *
from tqdm import tqdm
import matplotlib.pyplot as plt
class frame_recorder_dummy():

    def __init__(self, conf):
        pass

    def append_frame(self, sim):
        return
    
    def _window2frame(self, sim):
        return
    
    def export(self, export_name):
        return
    
    def reset(self):
        return
    
class frame_recorder(frame_recorder_dummy):
    def __init__(self, conf):
        self.conf = conf

        self._frames = []
        
    
    def append_frame(self, sim):
        pixels = sim.get_frame_from_renderer(cam_name='free_camera')
        self._frames.append(pixels)

    def reset(self):
        self._frames = []

    def export(self, export_name):
        if self._frames:
            
            
            if '.mp4' not in self.conf['export_path']:
                if exists_and_true('avoid_logfolder_names',self.conf):

                    export_path = self.conf['export_path']+'/'
                else:
                    export_path = self.conf['export_path']+'/'+self.conf['export_date_time']+'/'+export_name+'/'
            

                os.makedirs(export_path,exist_ok=True)
            else:
                export_path = self.conf['export_path']

            if exists_and_true('export_frames',self.conf):
                frames_path = export_path+'frames/'
                os.makedirs(frames_path,exist_ok=True)
                for i,frame in enumerate(tqdm(self._frames,desc='exporting frames',leave=False)):
                    frame = Image.fromarray(frame)
                    frame.save(frames_path+'/frame_'+str(i)+'.png')
            if '.mp4' not in self.conf['export_path']:
                if exists_and_true('export_video',self.conf):


                    media.write_video(
                                        export_path+'video.mp4',
                                        self._frames ,
                                        fps=self.conf['export_video_fps'],
                                        codec= 'hevc',
                                        )            
            else:

                if exists_and_true('export_video',self.conf):
                    # print(export_path)
                    media.write_video(
                                        export_path,
                                        self._frames ,
                                        fps=self.conf['export_video_fps'],
                                        codec= 'hevc',
                                        )

                # media.write_video(
                #                     './results/video_epi_'+str(n_epi)+'.mp4',
                #                     frames ,
                #                     fps=int(1/env.dt),
                #                     codec= 'hevc',
                #                 )
                # media.write_video(
                #                     export_path+'video.gif',
                #                     self._frames ,
                #                     fps=self.conf['export_video_fps'],
                #                     codec= 'gif',
                #                     )

if __name__ == "__main__":
    fr = frame_recorder(None)

    print(dir(fr))