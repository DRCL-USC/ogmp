import numpy as np
from preview_cmmp.utils import riccati_solve,riccati_inverse
class lqr:
    def __init__(self):
        pass
    def compute_params (self,a_mat,b_mat,Q,R):
        p_mat = riccati_solve(a_mat,b_mat,Q,R)
        inv = riccati_inverse(R,b_mat,p_mat)
        self.k_mat = inv@b_mat.T@p_mat
        
    def get_control(self,state_vector,state_ref):
        
        return  -self.k_mat@(state_vector-state_ref)

  
class lpc:
    def __init__(self):
        pass

    def compute_params(self,a_mat,b_mat,c_mat,Q,R,hor_len):

        n = len(a_mat)
        r = b_mat.shape[1] 
        p = len(c_mat)
        Qe = Q
        Qx = np.zeros((n,n))
        R = R
        Ip = np.identity(p)
        B_tilda = np.vstack((c_mat@b_mat,b_mat))
        F_tilda = np.vstack((c_mat@a_mat,a_mat))
        I_tilda  = np.vstack((Ip,np.zeros((n,p))))
        Q_tilda = np.vstack((np.hstack((Qe,np.zeros((p,n)))),np.hstack((np.zeros((n,p)),Qx))))
        A_tilda = np.hstack((I_tilda,F_tilda))
        K_tilda = riccati_solve(A_tilda,B_tilda,Q_tilda,R)
        inv = riccati_inverse(R,B_tilda,K_tilda)
        Ac_tilda = A_tilda- B_tilda@inv@B_tilda.T@K_tilda@A_tilda
        
        Gl = inv@B_tilda.T@K_tilda@I_tilda
        self.Gx = inv@B_tilda.T@K_tilda@F_tilda
        self.Gd = [None]*hor_len
        X_tilda = [None]*hor_len

        for i in range(hor_len):
            if i == 0:
                self.Gd[i] = -Gl
                X_tilda[i] = -Ac_tilda.T@K_tilda@I_tilda
            else:
                self.Gd[i] = inv@B_tilda.T@X_tilda[i-1]
                X_tilda[i] = Ac_tilda.T@X_tilda[i-1]


    def get_control(self,state_vector,output_ref):

        return - self.Gx@state_vector - np.array(sum([self.Gd[l]@output_ref[:,l] for l in range(max(1,output_ref.shape[1]-1))]).reshape((self.Gx.shape[0],1)))
    

