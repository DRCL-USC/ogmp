import os,sys
sys.path.append('./')
from oracles.lqr_orac import oracle_var
import numpy as np
import time 
if __name__ == "__main__":
    obj = oracle_var()
    x_init = np.array([[0.0, -0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,0.0, 0.0, -0.0, -9.8]]).T

    terrain_map = np.array(np.zeros((1,1500)))    

    '''fgb test'''
    start = [0,0.025,0.05,0.1,0.3,0.5,0.8]
    length = [0,0.1,0.2,0.3,0.4,0.5]
    height = [-0.2,0.0,0.1,0.2,0.3,0.4]
    internal_params = {'terrain_scan_x':terrain_map,'delta_tpos':np.array([[1.0],[0]]),'x0':x_init}
    for s in start:
        for l in length:
            for h in height:
                terrain_map[0,int(s*1000):int((s+l)*1000)] = h  
                X, U ,q_pos,q_vel = obj.get_traj('fbg',internal_params)
                terrain_map = np.array(np.zeros((1,1500))) 
                obj.plotter(X,U)

    # '''yaw_flip'''
    # yaw = [-2,-1,0,0.25,0.5,1,1.5,2]*np.pi
    # for y in yaw:
    #     internal_params = {'hi':0,'hf':0,'x0':x_init,"delta_apos":np.array([[0],[0],[y]])}
    #     X, U ,q_pos,q_vel = obj.get_traj('yaw_flip',internal_params)
    
    
    # t= time.time()
    # X, U ,q_pos,q_vel = obj.get_traj('yaw_flip',internal_params)
    # print(time.time()-t)
    # obj.plotter(X,U)
    # obj.animate(X,U)
  