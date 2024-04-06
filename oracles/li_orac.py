import os,sys
sys.path.append('./')
from preview_cmmp.oracle_base import oracle
from preview_cmmp.utils import square_wave_gen, W, convert2draw_vec
from preview_cmmp.draw import animate_traj,animate_traj_rot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


             


class oracle_var(oracle):
    def __init__(self, terrain_map_resolution=[0.001,0.001]):
        super().__init__()
        self._terrain_map_res = terrain_map_resolution
        

    def _index2coord_(self,i):
        return i*self._terrain_map_res[0]
    
    
    def _vz_calc(self,h1,h2,t):
        t = max(0.01,t)
        return (-self.g*t/2+(h2-h1)/t)/2
    
    
    
    def _terrain_map2list_(self,terrain_map):
        gap_blk_list = []
        pos_offset = 0.05
        ele_start = 0
        hi = 0
        ele_h = terrain_map[0][0]
        
        for i in range(len(terrain_map[0])) :
            if terrain_map[0][i] !=ele_h:

                if terrain_map[0][i]<0: #storing height before gap
                    hi = ele_h
                
                ele_end = self._index2coord_(i)  #giving an offset for the start of block

                if ele_h<0:  # storing the hi and hf also for gap
                    gap_blk_list.append([ele_start,ele_end,ele_h,hi,terrain_map[0][i]])
                else:  # for other two element only the start,end and height
                    gap_blk_list.append([ele_start,ele_end,ele_h])
                
                ele_start = ele_end
                ele_h = terrain_map[0][i]

        # storing the final element   
        if ele_h >=0:
            gap_blk_list.append([ele_start,self._index2coord_(len(terrain_map[0])),ele_h])
        else:
            gap_blk_list.append([ele_start,self._index2coord_(len(terrain_map[0])),ele_h,hi,0])
        
        return gap_blk_list
                

        # NEED TO DO: bring in new pseudo gap when jump needed between two blocks

    def _generate_reference_(self,mode_name,internal_params):

        if mode_name == "pitch_flip" or mode_name=="roll_flip":
            t_stance = 0.5
            w_nom = 8.95
            if (-self.g/2)*np.square(2*np.pi/w_nom-t_stance/2) >  abs(internal_params['hi']-internal_params['hf']):
                t_flight = 2*np.pi/w_nom - t_stance/2
            else:
                t_flight = np.sqrt(2*abs(internal_params['hi']-internal_params['hf'])/-self.g)

            self.tf = t_stance+t_flight
            v_ref = 1.0
            N = int(self.tf/self.ddt)
            state_ref = np.zeros((12,N+1))
            
            stance_iter = int((self.tf-t_flight)/self.ddt)

            ang_vel = np.sign(internal_params['delta_apos'])*2*np.pi/(0.5*(self.tf+1*t_flight))
            state_ref[6:9,:stance_iter] = np.linspace(np.zeros((3,1)),ang_vel,stance_iter).reshape(stance_iter,3).T
            state_ref[6:9,stance_iter:] = ang_vel
            state_ref[9,:] = np.sign(ang_vel[1])*v_ref
            state_ref[3,:] = np.linspace(0,np.sign(ang_vel[1][0])*v_ref*self.tf,N+1)+internal_params['x0'][3][0]
            state_ref[10,:] = -np.sign(ang_vel[0])*v_ref
            state_ref[4,:] = np.linspace(0,np.sign(ang_vel[0][0])*v_ref*self.tf,N+1)+internal_params['x0'][4][0]
            state_ref[5,:stance_iter] = internal_params['hi'] +0.5
            state_ref[5,stance_iter:] = np.linspace(internal_params['hi'],internal_params['hf'],N+1-stance_iter)+0.5
            
        
        elif mode_name =="yaw_flip":
            yaw_final = internal_params['delta_apos'][2][0]
            t_stance = 0.6*(abs(yaw_final)/(2*np.pi))
            w_nom = 8
            
            t_flight = abs(yaw_final)/w_nom - t_stance/2

            self.tf = t_stance+t_flight
            N = int(self.tf/self.ddt)
            state_ref = np.zeros((12,N+1))
            stance_iter = int((self.tf-t_flight)/self.ddt)

            ang_vel = yaw_final/(0.5*(self.tf+1*t_flight))
            state_ref[8,:stance_iter] = np.linspace(0,ang_vel,stance_iter)
            state_ref[8,stance_iter:] = ang_vel
            state_ref[5,:stance_iter] = internal_params['hi'] +0.5
            state_ref[5,stance_iter:] = np.linspace(internal_params['hi'],internal_params['hf'],N+1-stance_iter)+0.5
            
        elif mode_name == "fbg":
            gap_blk_list = self._terrain_map2list_(internal_params['terrain_scan_x'])
            self.tf = 1
            ref_vel = internal_params['delta_tpos']/self.tf
            N = int(self.tf/self.ddt)
            state_ref = np.zeros((12,N+1))
            v_ref_x= ref_vel[0][0]
            v_ref_y = ref_vel[1][0]
            state_ref[3,:] = np.linspace(0,v_ref_x*self.tf,N+1)+internal_params['x0'][3][0]
            state_ref[4,:] = np.linspace(0,v_ref_y*self.tf,N+1)+internal_params['x0'][4][0]
            state_ref[5,:] = 0.5
            
            self.gap_blk_list = gap_blk_list
            
            for i in range(len(gap_blk_list)):
                
                
                start_index = int(((gap_blk_list[i][0])/v_ref_x)/self.ddt)
                end_index = int(((gap_blk_list[i][1])/v_ref_x)/self.ddt)
                
                if gap_blk_list[i][2]>0:  #block
                    
                    state_ref[5,start_index:end_index] = gap_blk_list[i][2]+(0.5)  
                    state_ref[9,start_index:end_index] = v_ref_x                   
                    

                elif gap_blk_list[i][2]==0  : # flat
                    state_ref[5,start_index:end_index] = 0.5  
                    state_ref[9:11,start_index:end_index] = ref_vel                    
                   

                else: # gap
                    t = (gap_blk_list[i][1]- gap_blk_list[i][0])/v_ref_x
                    
                    H = -self.g*t*t/8
                    if end_index-int((start_index+end_index)/2) == int((end_index-start_index)/2):
                        state_ref[5,start_index:int((start_index+end_index)/2)] = np.linspace(0.5,H+0.5,int((end_index-start_index)/2))
                        state_ref[5,int((start_index+end_index)/2):end_index] = np.linspace(H+0.5,0.5,int((end_index-start_index)/2))
                    else:
                        state_ref[5,start_index:int((start_index+end_index)/2)+1] = np.linspace(0.5,H+0.5,int((end_index-start_index)/2)+1)
                        state_ref[5,int((start_index+end_index)/2):end_index] = np.linspace(H+0.5,0.5,int((end_index-start_index)/2)+1)
                     
                    state_ref[9,start_index:end_index] = v_ref_x    

        else :
            self.tf = 1
            N = int(self.tf/self.ddt)
            state_ref = np.zeros((12,N+1))
            state_ref[3,:] = internal_params['x0'][3][0]
            state_ref[4,:] = internal_params['x0'][4][0]
            state_ref[5,:] = 0.5
               

        return state_ref
    def get_traj(self, mode_name, input):
        state_ref= self._generate_reference_(mode_name,input)
        
        N = int(self.tf/self.ddt)
        x_sol = state_ref.copy()
        u_sol = np.zeros((18, N + 1)) 
    

        for i in range(N):
            x_sol[11][i] = (x_sol[5][i+1]-x_sol[5][i])/self.ddt
            x_sol[0:3,i+1] = x_sol[0:3,i]+x_sol[6:9,i]*self.ddt


           
        q_pos,q_vel = self._qpos_vel_traj_(x_sol)
        self.x_ref = state_ref.copy()
        return x_sol,u_sol,q_pos,q_vel

    def plotter(self,x_sol,u_sol):
        t = x_sol.shape[1]-1
        fig, axs = plt.subplots(2, 6,figsize=(10, 8))
        def format_y_ticks(y, pos):
            return f'{y:.2f}'
        
        state = {0:'roll',1:'pitch',2:'yaw',3:'x',4:'y',5:'z'}

        for i in range(6):
            axs[0,i].plot(x_sol[i,:t],label =state[i])
            axs[0,i].plot(self.x_ref[i,:t],label = state[i]+"_ref")
            axs[0,i].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))
            axs[1,i].plot(x_sol[i+6,:t],label =state[i])
            axs[1,i].plot(self.x_ref[i+6,:t],label = state[i]+"_ref")
            axs[1,i].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))
        
        for ax in axs.flat:
            ax.grid()
            ax.legend()
        plt.show()
        fig.tight_layout()
        plt.close(fig)

        fig, axs = plt.subplots(2, 3,figsize=(10, 8))
        control = {6:'flx',7:'fly',8:'flz',9:'frx',10:'fry',11:'frz',12:'tlx',13:'tly',14:'tlz',15:'trx',16:'try',17:'trz'}
        for i in range(3):
            axs[0,i].plot(u_sol[6+i,:t],label = control[6+i])
            axs[0,i].plot(u_sol[9+i,:t],label = control[9+i])
            axs[0,i].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))
            axs[1,i].plot(u_sol[12+i,:t],label = control[12+i])
            axs[1,i].plot(u_sol[15+i,:t],label = control[15+i])
            axs[1,i].yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))


        for ax in axs.flat:
            ax.grid()
            ax.legend()
        plt.show()
        fig.tight_layout()
        plt.close(fig)

    def animate(self,x_sol,u_sol,fname=None):
        
        animate_traj_rot(x_sol, u_sol,0, self.ddt,display=True,repeat = False)  # remove the fname argument if saving the video is not needed and it can directly be viewed.
        # animate_traj(x_sol,u_sol,self.gap_blk_list,self.ddt,display=True,repeat=False)



if __name__ == "__main__":
    obj = oracle_var()
    x_init = np.array([[0.0, -0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0,0.0, 0.0, -0.0, -9.8]]).T

    terrain_map = np.array(np.zeros((1,1000)))    
    
    terrain_map[0,205:500] = -0.3  #block
    # terrain_map[0,10:10]  = -0.3 #gap
    # internal_params = {'terrain_scan_x':terrain_map,'delta_tpos':np.array([[1],[0]]),'x0':x_init}
    internal_params = {'hi':2,'hf':0,'x0':x_init,"delta_apos":np.array([[0],[2*np.pi],[0]])}
    
    
    X, U ,q_pos,q_vel = obj.get_traj('stand',internal_params)
    obj.plotter(X,U)
    obj.animate(X,U)
    