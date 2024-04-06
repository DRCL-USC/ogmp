import numpy as np
from oracles.utils import rot_from_rpy

class biped_model:
    def __init__(self):
        # hector parameters
        self.body_inertia = np.diag([0.94 , 0.92, 0.069])
        self.m= 19.6
        self.ddt = 0.0125

    def _amat_(self,r,pi,y):
        a_mat = np.zeros((13,13))
        rrs = 0;rcs = 6 ; irs = 3 ; ics = 9
        R = np.matrix([[np.cos(pi)*np.cos(y),-np.sin(y),0],
                    [np.cos(pi)*np.sin(y),np.cos(y),0],
                    [0,0,1]])
        
        a_mat [rrs:rrs+3,rcs:rcs+3] = np.linalg.inv(R)
        a_mat [irs:irs+3,ics:ics+3] = np.identity(3)
        a_mat [11,12] = 1
        
        a_mat_dis = np.identity(13)+ a_mat*self.ddt
        return a_mat_dis
    
    def _bmat_(self,r,pi,y):
        R = rot_from_rpy(r,pi,y)
        inertia_world = R@self.body_inertia@R.T
        L = np.array([[1,0,0],[0,1,0],[0,0,1]])
        b_mat = np.zeros((6,12))
        angular_part = np.hstack((np.zeros((3,6)),np.linalg.inv(inertia_world)@L,np.linalg.inv(inertia_world)@L))
        linear_part = np.hstack(((1/self.m)*np.identity(3),(1/self.m)*np.identity(3),np.zeros((3,6))))
        b_mat = np.vstack((b_mat,angular_part,linear_part,np.zeros((1,12))))
        return b_mat*self.ddt
    
    def _fwdsim_(self,state_vector,control_vector,a_mat,b_mat):
        return a_mat@state_vector+ b_mat@control_vector