import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import sqrt

def create_layers(layer_fn, input_dim, layer_sizes):
  """
  This function creates a pytorch modulelist and appends
  pytorch modules like nn.Linear or nn.LSTMCell passed
  in through the layer_fn argument, using the sizes
  specified in the layer_sizes list.
  """
  ret = nn.ModuleList()
  ret += [layer_fn(input_dim, layer_sizes[0])]
  for i in range(len(layer_sizes)-1):
    ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
  return ret

class Net(nn.Module):
  """
  The base class which all policy networks inherit from. It includes methods
  for normalizing states.
  """
  def __init__(self):
    super(Net, self).__init__()
    #nn.Module.__init__(self)
    self.is_recurrent = False

    self.state_mean = torch.zeros(1)
    self.state_mean_diff = torch.ones(1)
    self.state_n = 1

    self.env_name = None

    self.calculate_norm = False

  def normalize_state(self, state, update=True):
    """
    Use Welford's algorithm to normalize a state, and optionally update the statistics
    for normalizing states using the new state, online.
    """
    state = torch.Tensor(state)

    if self.state_n == 1:
      self.state_mean = torch.zeros(state.size(-1))
      self.state_mean_diff = torch.ones(state.size(-1))

    if update:
      if len(state.size()) == 1: # if we get a single state vector 
        state_old = self.state_mean
        self.state_mean += (state - state_old) / self.state_n
        self.state_mean_diff += (state - state_old) * (state - state_old)
        self.state_n += 1
      else:
        raise RuntimeError # this really should not happen
    return (state - self.state_mean) / sqrt(self.state_mean_diff / self.state_n)

  def copy_normalizer_stats(self, net):
    self.state_mean      = net.state_mean
    self.state_mean_diff = net.state_mean_diff
    self.state_n         = net.state_n
  
class FF_Base(Net):
  """
  The base class for feedforward networks.
  """
  def __init__(self, in_dim, layers, nonlinearity):
    super(FF_Base, self).__init__()
    self.layers       = create_layers(nn.Linear, in_dim, layers)
    self.nonlinearity = nonlinearity

  def _base_forward(self, x):
    for idx, layer in enumerate(self.layers):
      x = self.nonlinearity(layer(x))
    return x

class LSTM_Base(Net):
  """
  The base class for LSTM networks.
  """
  def __init__(self, in_dim, layers):
    super(LSTM_Base, self).__init__()
    self.layers = create_layers(nn.LSTMCell, in_dim, layers)

  def init_hidden_state(self, batch_size=1):
    self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]
    self.cells  = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

  def _base_forward(self, x):
    dims = len(x.size())

    if dims == 3: # if we get a batch of trajectories
      self.init_hidden_state(batch_size=x.size(1))

      if self.calculate_norm:
        self.latent_norm = 0

      y = []
      for t, x_t in enumerate(x):
        for idx, layer in enumerate(self.layers):
          c, h = self.cells[idx], self.hidden[idx]
          self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
          x_t = self.hidden[idx]

          if self.calculate_norm:
            self.latent_norm += (torch.mean(torch.abs(x_t)) + torch.mean(torch.abs(self.cells[idx])))

        y.append(x_t)
      x = torch.stack([x_t for x_t in y])

      if self.calculate_norm:
        self.latent_norm /= len(x) * len(self.layers)

    else:
      if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
        x = x.view(1, -1)

      for idx, layer in enumerate(self.layers):
        h, c = self.hidden[idx], self.cells[idx]
        self.hidden[idx], self.cells[idx] = layer(x, (h, c))
        x = self.hidden[idx]

      if dims == 1:
        x = x.view(-1)
    return x
