"""
Contains the code implementing Proximal Policy Optimization (PPO),
with objective clipping and early termination if a KL threshold 
is exceeded.
"""

import os
import ray
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from time import time
from tqdm import tqdm
import yaml
from timeit import default_timer as timer

class Buffer:
  """
  A class representing the replay buffer used in PPO. Used
  to states, actions, and reward/return, and then calculate
  advantage from discounted sum of returns.
  """
  def __init__(self, discount=0.99):
    self.discount = discount
    self.states     = []
    self.actions    = []
    self.rewards    = []
    self.values     = []
    self.returns    = []
    self.advantages = []

    self.size = 0

    self.traj_idx = [0]
    self.buffer_ready = False

  def __len__(self):
    return len(self.states)

  def push(self, state, action, reward, value, done=False):
    self.states  += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values  += [value]

    self.size += 1

  def end_trajectory(self, terminal_value=0):
    self.traj_idx += [self.size]
    rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

    returns = []

    R = terminal_value
    for reward in reversed(rewards):
        R = self.discount * R + reward
        returns.insert(0, R)

    self.returns += returns

  def _finish_buffer(self):

    self.states  = torch.Tensor(np.array(self.states))
    self.actions = torch.Tensor(np.array(self.actions))
    self.rewards = torch.Tensor(np.array(self.rewards))
    self.returns = torch.Tensor(np.array(self.returns))
    self.values  = torch.Tensor(np.array(self.values))

    a = self.returns - self.values
    a = (a - a.mean()) / (a.std() + 1e-4)
    self.advantages = a
    self.buffer_ready = True

  def sample(self, batch_size=64, recurrent=False):
    if not self.buffer_ready:
      self._finish_buffer()

    if recurrent:
      """
      If we are returning a sample for a recurrent network, we should
      return a zero-padded tensor of size [traj_len, batch_size, dim],
      or a trajectory of batched states/actions/returns.
      """
      random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for traj_indices in sampler:
        states     = [self.states[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
        actions    = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
        returns    = [self.returns[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
        advantages = [self.advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]

        traj_mask  = [torch.ones_like(r) for r in returns]
        
        states     = pad_sequence(states, batch_first=False)
        actions    = pad_sequence(actions, batch_first=False)
        returns    = pad_sequence(returns, batch_first=False)
        advantages = pad_sequence(advantages, batch_first=False)
        traj_mask  = pad_sequence(traj_mask, batch_first=False)

        yield states, actions, returns, advantages, traj_mask

    else:
      """
      If we are returning a sample for a conventional network, we should
      return a tensor of size [batch_size, dim], or a batch of timesteps.
      """
      random_indices = SubsetRandomSampler(range(self.size))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for i, idxs in enumerate(sampler):
        states     = self.states[idxs]
        actions    = self.actions[idxs]
        returns    = self.returns[idxs]
        advantages = self.advantages[idxs]

        yield states, actions, returns, advantages, 1

@ray.remote
class PPO_Worker:
  """
  A class representing a parallel worker used to explore the
  environment.
  """
  def __init__(self, actor, critic, env_fn, gamma):
    torch.set_num_threads(1)
    self.gamma = gamma
    self.actor = deepcopy(actor)
    self.critic = deepcopy(critic)
    self.env = env_fn()

  def sync_policy(self, new_actor_params, new_critic_params, input_norm=None):
    for p, new_p in zip(self.actor.parameters(), new_actor_params):
      p.data.copy_(new_p)

    for p, new_p in zip(self.critic.parameters(), new_critic_params):
      p.data.copy_(new_p)

    if input_norm is not None:
      self.actor.state_mean, self.actor.state_mean_diff, self.actor.state_n = input_norm

  def collect_experience(self, max_traj_len, min_steps):
    with torch.no_grad():
      start = time()

      num_steps = 0
      memory = Buffer(self.gamma)
      actor  = self.actor
      critic = self.critic

      while num_steps < min_steps:
        state = torch.Tensor(self.env.reset())

        done = False
        value = 0
        traj_len = 0

        if hasattr(actor, 'init_hidden_state'):
          actor.init_hidden_state()

        if hasattr(critic, 'init_hidden_state'):
          critic.init_hidden_state()

        # while not done and traj_len < max_traj_len:
        while not done:

            state = torch.Tensor(state)
            action = actor(state, False)
            value = critic(state)

            next_state, reward, done, _ = self.env.step(action.numpy())

            reward = np.array([reward])

            memory.push(state.numpy(), action.numpy(), reward, value.numpy())

            state = next_state

            traj_len += 1
            num_steps += 1

        value = (not done) * critic(torch.Tensor(state)).numpy()
        memory.end_trajectory(terminal_value=value)

      return memory

class PPO:
    def __init__(self, actor, critic, env_fn, args):

      self.actor = actor
      self.old_actor = deepcopy(actor)
      self.critic = critic


      if actor.is_recurrent or critic.is_recurrent:
        self.recurrent = True
      else:
        self.recurrent = False

      self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.a_lr, eps=args.eps)
      self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.c_lr, eps=args.eps)
      self.env_fn = env_fn
      self.discount = args.discount
      self.grad_clip = args.grad_clip

      if not ray.is_initialized():
        ray.init(num_cpus=args.workers)

      self.workers = [PPO_Worker.remote(actor, critic, env_fn, args.discount) for _ in range(args.workers)]

    def sync_policy(self, states, actions, returns, advantages, mask):
      with torch.no_grad():
        old_pdf       = self.old_actor.pdf(states)
        old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)

      pdf        = self.actor.pdf(states)
      log_probs  = pdf.log_prob(actions).sum(-1, keepdim=True)

      ratio      = ((log_probs - old_log_probs)).exp()
      cpi_loss   = ratio * advantages * mask
      clip_loss  = ratio.clamp(0.8, 1.2) * advantages * mask
      actor_loss = -torch.min(cpi_loss, clip_loss).mean()

      critic_loss = 0.5 * ((returns - self.critic(states)) * mask).pow(2).mean()

      self.actor_optim.zero_grad()
      self.critic_optim.zero_grad()

      actor_loss.backward()
      critic_loss.backward()

      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
      self.actor_optim.step()
      self.critic_optim.step()

      with torch.no_grad():
        return kl_divergence(pdf, old_pdf).mean().numpy(), ((actor_loss).item(), critic_loss.item())

    def merge_buffers(self, buffers):
      memory = Buffer()

      for b in buffers:
        offset = len(memory)

        memory.states  += b.states
        memory.actions += b.actions
        memory.rewards += b.rewards
        memory.values  += b.values
        memory.returns += b.returns

        memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
        memory.size     += b.size
      return memory

    def do_iteration(self, num_steps, max_traj_len, epochs, kl_thresh=0.02, verbose=True, batch_size=64):
      self.old_actor.load_state_dict(self.actor.state_dict())

      start = time()
      actor_param_id  = ray.put(list(self.actor.parameters()))
      critic_param_id = ray.put(list(self.critic.parameters()))
      norm_id = ray.put([self.actor.state_mean, self.actor.state_mean_diff, self.actor.state_n])

      steps = max(num_steps // len(self.workers), max_traj_len)

      for w in self.workers:
        w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)

      if verbose:
        print("\t{:5.4f}s to copy policy params to workers.".format(time() - start))

      start = time()
      buffers = ray.get([w.collect_experience.remote(max_traj_len, steps) for w in self.workers])
      memory = self.merge_buffers(buffers)

      total_steps = len(memory)
      elapsed = time() - start
      if verbose:
        print("\t{:3.2f}s to collect {:6n} timesteps | {:3.2}k/s.".format(elapsed, total_steps, (total_steps/1000)/elapsed))

      start  = time()
      kls    = []
      done = False
      for epoch in range(epochs):
        a_loss = []
        c_loss = []
        for batch in memory.sample(batch_size=batch_size, recurrent=self.recurrent):
          states, actions, returns, advantages, mask = batch
          
          kl, losses = self.sync_policy(states, actions, returns, advantages, mask)
          kls += [kl]
          a_loss += [losses[0]]
          c_loss += [losses[1]]

          if max(kls) > kl_thresh:
              done = True
              print("\t\tbatch had kl of {} (threshold {}), stopping optimization early.".format(max(kls), kl_thresh))
              break

        if verbose:
          print("\t\tepoch {:2d} kl {:4.3f}, actor loss {:6.3f}, critic loss {:6.3f}".format(epoch+1, np.mean(kls), np.mean(a_loss), np.mean(c_loss)))

        if done:
          break

      if verbose:
        print("\t{:3.2f}s to update policy.".format(time() - start))
      return np.mean(kls), np.mean(a_loss), np.mean(c_loss), len(memory)

def run_experiment(args,**kwargs):
    torch.set_num_threads(1)

    from src.util import create_logger, env_factory, eval_policy_trng, train_normalizer

    from nn.critic import FF_V, LSTM_V
    from nn.actor import FF_Stochastic_Actor, LSTM_Stochastic_Actor

    import locale, os
    locale.setlocale(locale.LC_ALL, '')


    # create a tensorboard logging object
    if not args.nolog:
      logger = create_logger(args)
    else:
      logger = None


    # wrapper function for creating parallelized envs
    env_fn     = env_factory(
                              args.exp_conf_path
                            )

    obs_dim    = env_fn().observation_space.shape[0]    
    action_dim = env_fn().action_space.shape[0]
    layers     = [int(x) for x in args.layers.split(',')]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.load_initial_agent_from == None:
      if args.recurrent:
        policy = LSTM_Stochastic_Actor(
                                        obs_dim, 
                                        action_dim,
                                        layers=layers, 
                                        dynamics_randomization=args.randomize, 
                                        fixed_std=torch.ones(action_dim)*args.std
                                      )
        critic = LSTM_V(obs_dim, layers=layers)
      else:
        policy = FF_Stochastic_Actor(obs_dim, action_dim,\
                                    layers=layers, 
                                    dynamics_randomization=args.randomize, 
                                    fixed_std=torch.ones(action_dim)*args.std)
        critic = FF_V(obs_dim, layers=layers)
    elif isinstance(args.load_initial_agent_from,str):
      print('\nloading initial ppolicy and critic from:',args.load_initial_agent_from)
      policy = torch.load(os.path.join(args.load_initial_agent_from,'actor.pt'))
      critic = torch.load(os.path.join(args.load_initial_agent_from,'critic.pt'))
    else:
      print('incompatible datatype for args.load_initial_agent_from, enter a path')
      exit()

    env = env_fn()

    # lok-i, 17/9/22
    # policy.train(0)
    # critic.train(0)
    policy.train(False)
    critic.train(False)



    
    print("Collecting normalization statistics with {} states...".format(args.prenormalize_steps))
    train_normalizer(
                      policy, 
                      args.prenormalize_steps, 
                      max_traj_len=args.traj_len, 
                      noise=1,
                      exp_conf_path=args.exp_conf_path
                    )
    critic.copy_normalizer_stats(policy)

    algo = PPO(policy, critic, env_fn, args)

    if args.save_actor is None and logger is not None:
      args.save_actor = os.path.join(logger.dir, 'actor.pt')

    if args.save_critic is None and logger is not None:
      args.save_critic = os.path.join(logger.dir, 'critic.pt')


    print()
    print("Proximal Policy Optimization:")
    print("\tseed:               {}".format(args.seed))
    print("\ttimesteps:          {:n}".format(int(args.timesteps)))
    print("\titeration steps:    {:n}".format(int(args.sample)))
    print("\tprenormalize steps: {}".format(int(args.prenormalize_steps)))
    print("\ttraj_len:           {}".format(args.traj_len))
    print("\tdiscount:           {}".format(args.discount))
    print("\tactor_lr:           {}".format(args.a_lr))
    print("\tcritic_lr:          {}".format(args.c_lr))
    print("\tgrad clip:          {}".format(args.grad_clip))
    print("\tbatch size:         {}".format(args.batch_size))
    print("\tepochs:             {}".format(args.epochs))
    print("\trecurrent:          {}".format(args.recurrent))
    print("\tdynamics rand:      {}".format(args.randomize))
    print("\tworkers:            {}".format(args.workers))
    print("\tmax_itr:                {}".format(args.max_itr))
    conf_file = open(args.exp_conf_path)
    exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)
    print('\texperiment configuration {}'.format(exp_conf))    
    print()

    itr = 0
    timesteps = 0
    best_reward = None

    
    while timesteps < args.timesteps and itr < args.max_itr:
      
      print('######################### itr:',itr,'#########################')
      start = timer()
      kl, a_loss, c_loss, steps = algo.do_iteration(
                                                      args.sample, 
                                                      args.traj_len, 
                                                      args.epochs, 
                                                      batch_size=args.batch_size, 
                                                      kl_thresh=args.kl
                                                    )
      end = timer()
      print('time taken for iteration: ',end-start)

      start = timer()
      eval_reward = eval_policy_trng(
                                  algo.actor, 
                                  env, 
                                  episodes=5, 
                                  max_traj_len=args.traj_len, 
                                  verbose=False,
                                  iteration = itr 
                                  )
      end = timer()
      print('time taken for evaluation: ',end-start)


      timesteps += steps


      print("iter {:4d} | return: {:5.2f} | KL {:5.4f} | timesteps {:n}".format(itr, eval_reward, kl, timesteps))

      if best_reward is None or eval_reward > best_reward:
        print("\t(best policy so far! saving to {})".format(args.save_actor))
        best_reward = eval_reward
        if args.save_actor is not None:
          torch.save(algo.actor, args.save_actor)
        
        if args.save_critic is not None:
          torch.save(algo.critic, args.save_critic)

      if logger is not None:
        logger.add_scalar('ppo/kl', kl, itr)
        logger.add_scalar('ppo/return', eval_reward, itr)
        logger.add_scalar('ppo/actor_loss', a_loss, itr)
        logger.add_scalar('ppo/critic_loss', c_loss, itr)
      itr += 1
    print("Finished ({} of {}).".format(timesteps, args.timesteps))
