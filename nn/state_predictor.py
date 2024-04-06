from typing import Iterator
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class StatePredictor(nn.Module):
    def __init__(
                    self,
                    x0,
                    obs_dim, 
                    action_dim,
                    network_arch = [64,64], 
                    activation = nn.ReLU,
                    dt = 0.03,
                    oos = 'residual'
                ):
        super(StatePredictor, self).__init__()
        self.dt = dt
        self.x0 = x0
        self.x_minus = None
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = x0.shape[0]
        self.network_arch = network_arch
        self.activation = activation

        self.oos = getattr(self, f'oos_{oos}')
        self.model = self.build_model()
    
    # output override scheme
    def oos_resd_intg(self,x,x_minus):
        x[:,0:6] = x[:,0:6] + x_minus[:,0:6] + x_minus[:,6:12]*self.dt
        return x
    
    def oos_pure_intg(self,x,x_minus):
        x[:,0:6] = x_minus[:,0:6] + x_minus[:,6:12]*self.dt
        return x

    def oos_none(self,x,x_minus):
        return x

    def reset_x_minus(self,x0=None):
        if x0 is not None:
            self.x0 = x0
        self.x_minus = self.x0

    def build_model(self):
        layers = []
        layers.append(nn.Linear(self.obs_dim+self.action_dim+self.state_dim, self.network_arch[0]))
        layers.append(self.activation())
        for i in range(1,len(self.network_arch)):
            layers.append(nn.Linear(self.network_arch[i-1], self.network_arch[i]))
            layers.append(self.activation())
        layers.append(nn.Linear(self.network_arch[-1], self.state_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def predict(self, obss, acts):

        n_trajs = obss.shape[0]
        n_steps = obss.shape[1]

        # forward simulate multiple trajs in parallel, gradient is not affected
        pred = torch.zeros(n_trajs, n_steps, self.state_dim).to(obss.device)
        if self.x_minus.shape[0] != n_trajs:
            self.x_minus = self.x_minus.repeat(n_trajs,1)
        for i in range(n_steps):
            x = torch.cat([obss[:,i,:],acts[:,i,:],self.x_minus],dim=-1)
            x = self.forward(x)
            x = self.oos(x,self.x_minus)
            pred[:,i,:] = x
            self.x_minus = x

        return pred



if __name__ == "__main__":
    obs_dim = 48
    action_dim = 10
    state_dim = 12+2 
    model = StatePredictor(obs_dim, action_dim, state_dim)
    print(model)
    obs = torch.rand(1, obs_dim)
    action = torch.rand(1, action_dim)
    print(model(obs, action))