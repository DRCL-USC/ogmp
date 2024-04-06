import os,sys
sys.path.append('./')
from preview_cmmp.oracle_base import oracle
from preview_cmmp.optimisers import lqr
from preview_cmmp.utils import square_wave_gen, W, convert2draw_vec
from preview_cmmp.draw import animate_traj,animate_traj_rot
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


              #z,wr,wp,wy,vx,vy,vz
Q_z = np.diag([0,0,0,0,0,11000,1,1,1,10,3,0,0])
Q_vz =np.diag([0,0,0,0,0,0,1,1,1,1,3,1000,0])
R = np.diag([1e-4,1e-5,1e-3,1e-4,1e-5,1e-3,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4])
T_cycle_nom = 0.25
H_nom = 25
Oplen =7

class oracle_var(oracle):
    def __init__(self, terrain_map_resolution=[0.001,0.001]):
        super().__init__()
        self._terrain_map_res = terrain_map_resolution
        

    def _index2coord_(self,i):
        return i*self._terrain_map_res[0]
    
    def _con_seq_gen_(self,start_index,end_index):      ## function to find the contact sequence : double stance and flight
        N = end_index-start_index  
        t_ele = N*self.ddt
        c = max(1,math.ceil((t_ele/T_cycle_nom)))
        f = max(1,math.floor((t_ele/T_cycle_nom)))
        n = c if abs(t_ele/c-T_cycle_nom)<abs(t_ele/f-T_cycle_nom) else f
        
        t_cycle  = t_ele/n
        if t_cycle>T_cycle_nom+0.1:
            t_cycle = t_cycle/2
        
        if t_cycle==0:
            return 1,[0] 
        n = int(t_ele/t_cycle)     # number of cycles in the given element. 
        
        return square_wave_gen(n,N), [0.5*i*N/n for i in range(2*n)] #sending the square wave as contact ref and the list of switch iteration number

    def _vz_calc(self,h1,h2,t):
        t = max(0.01,t)
        return (-self.g*t/2+(h2-h1)/t)/2
    
    def _q_selector(self,enc):
        if enc == 0:
            return Q_z
        else:
            return Q_vz
    
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
                
                ele_end = max(self._index2coord_(i) - int(terrain_map[0][i]>0 and terrain_map[0][i]>ele_h)*pos_offset,0) #giving an offset for the start of block

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
            N = int(1.5*self.tf/self.ddt)
            state_ref = np.zeros((13,N))
            contact_ref = np.zeros(N)
            self.Q_ref = np.ones(N+1)
            stance_iter = int((self.tf-t_flight)/self.ddt)

            ang_vel = np.sign(internal_params['delta_apos'])*2*np.pi/(0.5*(self.tf+1.0*t_flight))
            state_ref[0:3,:] = np.linspace(np.zeros((3,1)),np.sign(internal_params['delta_apos'])*3*np.pi,N).reshape(N,3).T
            state_ref[6:9,:stance_iter] = np.linspace(np.zeros((3,1)),ang_vel,stance_iter).reshape(stance_iter,3).T
            state_ref[6:9,stance_iter:] = ang_vel
            state_ref[9,:] = np.sign(ang_vel[1])*v_ref
            state_ref[10,:] = -np.sign(ang_vel[0])*v_ref
            state_ref[11,:] = -self.g*t_flight/2 -(internal_params['hi']-internal_params['hf'])/t_flight
            contact_ref[:stance_iter] = 1
            switch_iter_list = [0,stance_iter,N-1]
        
        elif mode_name =="yaw_flip":
            yaw_final = internal_params['delta_apos'][2][0]
            t_stance = 0.6*(abs(yaw_final)/(2*np.pi))
            w_nom = 8
            
            t_flight = abs(yaw_final)/w_nom - t_stance/2

            self.tf = t_stance+t_flight
            N = int(1.5*self.tf/self.ddt)
            state_ref = np.zeros((13,N))
            contact_ref = np.zeros(N)
            self.Q_ref = np.ones(N+1)
            stance_iter = int((self.tf-t_flight)/self.ddt)

            ang_vel = yaw_final/(0.5*(self.tf+1.05*t_flight))
            state_ref[2,:] = np.linspace(0,1.5*yaw_final,N)
            state_ref[8,:stance_iter] = np.linspace(0,ang_vel,stance_iter)
            state_ref[8,stance_iter:] = ang_vel
            vert_vel = 0.5*(-self.g*t_flight/2 -(internal_params['hi']-internal_params['hf'])/t_flight)
            state_ref[11,:int(stance_iter/2)] = np.linspace(0,vert_vel,int(stance_iter/2))
            state_ref[11,int(stance_iter/2):] = vert_vel 
            contact_ref[:stance_iter] = 1
            switch_iter_list = [0,stance_iter,N-1]

        elif mode_name == "fbg":
            gap_blk_list = self._terrain_map2list_(internal_params['terrain_scan_x'])
            self.tf = 1
            ref_vel = internal_params['delta_tpos']/self.tf
            N = int(1.5*self.tf/self.ddt)
            state_ref = np.zeros((13,N))
            v_ref_x= ref_vel[0][0]
            v_ref_y = ref_vel[1][0]
            state_ref[3,:] = np.linspace(0,1.5*v_ref_x*self.tf,N)+internal_params['x0'][3][0]
            state_ref[4,:] = np.linspace(0,1.5*v_ref_y*self.tf,N)+internal_params['x0'][4][0]
            state_ref[5,:] = 0.55
            contact_ref = np.zeros(N)
            self.Q_ref = np.zeros(N+1)
            switch_iter_list = []
            self.gap_blk_list = gap_blk_list
            
            for i in range(len(gap_blk_list)):
                
                
                start_index = int(((gap_blk_list[i][0])/v_ref_x)/self.ddt)
                end_index = int(((gap_blk_list[i][1])/v_ref_x)/self.ddt)
                contact_ref[start_index:end_index],local_switch_iter_list = self._con_seq_gen_(start_index,end_index)

                if gap_blk_list[i][2]>0:  #block
                    end_index_wofst = int(((gap_blk_list[i][1]+0.2)/v_ref_x)/self.ddt)
                    state_ref[5,start_index:end_index_wofst] = gap_blk_list[i][2]+(0.55)  
                    state_ref[9,start_index:end_index_wofst] = v_ref_x                   
                    self.Q_ref[start_index:end_index] = 0    

                elif gap_blk_list[i][2]==0  : # flat
                    state_ref[5,start_index:end_index] = 0.55  
                    state_ref[9:11,start_index:end_index] = ref_vel                    
                    self.Q_ref[start_index:end_index] = 0 

                 
                else: # gap
                    t = (gap_blk_list[i][1]- gap_blk_list[i][0])/v_ref_x
                    # end_index_wofst = int(((gap_blk_list[i][1])/v_ref_x)/self.ddt)
                    state_ref[11,start_index:end_index] = self._vz_calc(gap_blk_list[i][3],gap_blk_list[i][4],t)
                    state_ref[5,start_index:end_index] = 0.55  
                    state_ref[9,start_index:end_index] = v_ref_x                   
                    self.Q_ref[start_index:end_index] = 1  
                    contact_ref[start_index:end_index] = square_wave_gen(1,(end_index-start_index))
                    local_switch_iter_list = [0,int(0.5*(-start_index+end_index))]

                
                switch_iter_list+=map(lambda x: x + start_index, local_switch_iter_list)
            switch_iter_list.append(end_index)
            switch_iter_list=list(map(int, switch_iter_list))
        
        else :
            self.tf = 1
            N = int(1.5*self.tf/self.ddt)
            state_ref = np.zeros((13,N+1))
            self.Q_ref = np.zeros(N+1)
            contact_ref = np.ones(N+1)
            switch_iter_list = [0,N]
            state_ref[3,:] = internal_params['x0'][3][0]
            state_ref[4,:] = internal_params['x0'][4][0]
            state_ref[5,:] = 0.5

        state_ref[12,:] = self.g  
        return state_ref,contact_ref,switch_iter_list
        
    def get_traj(self, mode_name, input):
        state_ref, contact_ref,switch_iter_list = self._generate_reference_(mode_name,input)
        state_vector  = input['x0'].copy()

        
            
        
        N = int(self.tf/self.ddt)
        x_sol = np.zeros((12, N + 1))
        u_sol = np.zeros((18, N + 1)) 
        lqr_obj = lqr()

        foot_pos = [np.array([state_vector[3][0], W / 2.0,0.0]),np.array([state_vector[3][0], -W / 2.0,0.0])]
        
        control_vector_sw = np.zeros((12,1))
        i = 0
        for k in range(N+1):

            if k :
                if bool(contact_ref[k])^bool(contact_ref[k-1]) :      # condition for switch 
                    r_avg = 0.5*(state_ref[0,switch_iter_list[i]]+state_ref[0,switch_iter_list[i+1]])
                    pi_avg = 0.5*(state_ref[1,switch_iter_list[i]]+state_ref[1,switch_iter_list[i+1]])
                    y_avg = 0.5*(state_ref[2,switch_iter_list[i]]+state_ref[2,switch_iter_list[i+1]])
                    a_mat = self.model._amat_(r_avg,pi_avg,y_avg)
                    b_mat = self.model._bmat_(r_avg,pi_avg,y_avg)
                    q_mat = self._q_selector(self.Q_ref[k])
                    lqr_obj.compute_params(a_mat,b_mat,q_mat,R)
                    i+=1
                
            else :  # first iteration alone
                r_avg = 0.5*(state_ref[0,switch_iter_list[i]]+state_ref[0,switch_iter_list[i+1]])
                pi_avg = 0.5*(state_ref[1,switch_iter_list[i]]+state_ref[1,switch_iter_list[i+1]])
                y_avg = 0.5*(state_ref[2,switch_iter_list[i]]+state_ref[2,switch_iter_list[i+1]])
                a_mat = self.model._amat_(r_avg,pi_avg,y_avg)
                b_mat = self.model._bmat_(r_avg,pi_avg,y_avg)
                q_mat = self._q_selector(self.Q_ref[k])
                lqr_obj.compute_params(a_mat,b_mat,q_mat,R)
                i+=1
                

            #stance phase 
            if contact_ref[k] ==1:
                control_vector = lqr_obj.get_control(state_vector,state_ref[:,k].reshape((13,1)))
                
            
            #flight
            else:
                control_vector = control_vector_sw

            # print(state_vector,k,control_vector)
            state_vector = self.model._fwdsim_(state_vector,control_vector,a_mat,b_mat)
            
            x_sol[:,k], u_sol[:,k]  = convert2draw_vec(state_vector,foot_pos,control_vector)
           
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
    x_init = np.array([[0.0, -0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,0.0, 0.0, -0.0, -9.8]]).T

    terrain_map = np.array(np.zeros((1,1000)))    
    
    terrain_map[0,205:700] = -0.3  #block
    # terrain_map[0,10:10]  = -0.3 #gap
    internal_params = {'terrain_scan_x':terrain_map,'delta_tpos':np.array([[1],[0]]),'x0':x_init}
    # internal_params = {'hi':2,'hf':0.0,'x0':x_init,"delta_apos":np.array([[0],[2*np.pi],[0]])}
    
    X, U ,q_pos,q_vel = obj.get_traj('stand',internal_params)
    obj.plotter(X,U)
    obj.animate(X,U)
    