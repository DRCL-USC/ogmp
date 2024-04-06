
from torch.autograd import Function, Variable
from qpth.qp import QPFunction
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch


class SrbQpCtrl(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Torch implementation of the SRB-QP controller

        the optimal control problem

        state vector = [qpos, qvel] = [p, θ, j, dp, dθ, dj]

        given: ddq_base_des, c_des and feedback 

        λ* = argmin: ||ddq_base - ddq_base_des||_R1^2 + ||λ - λ_des||_R2^2
        
             subject to:
                            [ddp, ddθ] = M^-1 * (λ - C)
                            
                            λ ∈ λ_contact_force 
                            
                            τ ∈ [τ_min, τ_max]

        '''
        
        self.lamda_dim = 10
        self.R1 = torch.eye(self.lamda_dim) # weight matrix for ddq_base term
        self.R2 = torch.eye(self.lamda_dim) # weight matrix for lambda term


        pass

    def forward(
                self, 
                x,
                ):
        '''
            x = [qpos, qvel, qacc_des, c_des]
        '''


        Q, q, G, h, A, b = self._get_cost_constraints(x)
        
        x = QPFunction(verbose=-1)(
                                        Q, 
                                        q, 
                                        G, 
                                        h, 
                                        A,
                                        b,
                                   )        
        x = self._get_torque_from_lambda(x).float()
        return x
    
    def _get_cost_constraints(self, x):
        

        qpos = x[0:16]
        qvel = x[16:32]
        qacc_des = x[32:38]
        c_des = x[38:44]

        base_pos = qpos[0:3]
        base_rpy = qpos[3:6]
        jpos = qpos[6:]

        base_tvel = qvel[0:3]
        base_avel = qvel[3:6]
        jvel = qvel[6:]
        
        # print('###############')
        # print("base_pos:", base_pos, base_pos.shape)
        # print("base_rpy:", base_rpy, base_rpy.shape)
        # print("jpos:", jpos, jpos.shape)
        # print("base_tvel:", base_tvel, base_tvel.shape)
        # print("base_avel:", base_avel, base_avel.shape)
        # print("jvel:", jvel, jvel.shape)
        # print("qacc_des:", qacc_des, qacc_des.shape)
        # print("c_des:", c_des, c_des.shape)



        '''
        cost function: z^T * Q * z + q^T * z
 
            
            compute the dynamics matrices in the form of L and k
            ddq = L * λ + k

            Q = L^T * R1 * L + R2* I
            q = -L^T * R1 * k 
        
        constraints: Gz <= h, Az = b
            
            inequality constraints: G and h matrices 
            equality constraints: A and b matrices

        '''
        Q = torch.eye(self.lamda_dim).double()
        q = torch.zeros(self.lamda_dim).double()

        # unassigned for now
        G = Variable(-torch.eye(self.lamda_dim)).double()
        h = Variable(torch.zeros(self.lamda_dim)).double()
        A = Variable(torch.Tensor()).double()
        b = Variable(torch.Tensor()).double()

        return Q, q, G, h, A, b

    def _get_foot_jacobian(self):
        pass

    def _get_torque_from_lambda(self, _lambda):
        tau = _lambda
        return tau
