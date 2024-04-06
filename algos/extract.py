import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import locale, os, time

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from nn.fit import Model
from util import env_factory

# NOTE: 1/10/22
# yet to update env_factory(conf_path) updates... currenty code broken

def get_hiddens(policy):
  """
  A helper function for flattening the memory of a recurrent
  policy into a vector.
  """
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += [h.data for h in policy.hidden]
  
  if hasattr(policy, 'cells'):
    hiddens += [c.data for c in policy.cells]

  if hasattr(policy, 'latent'):
    hiddens += [l for l in policy.latent]
  
  return torch.cat([layer.view(-1) for layer in hiddens]).numpy()
  
def collect_point(policy, max_traj_len,exp_conf_path):
  """
  A helper function which collects a single memory-dynamics parameter pair
  from a trajectory.
  """
  env = env_factory(exp_conf_path)()

  chosen_timestep = np.random.randint(15, max_traj_len)
  timesteps = 0
  done = False

  if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

  state = env.reset()
  while not done and timesteps < chosen_timestep:

    action = policy(state).numpy()
    state, _, done, _ = env.step(action)
    timesteps += 1

  return get_hiddens(policy), env.get_damping(), env.get_mass(), env.get_ipos()

@ray.remote
def collect_data(policy, max_traj_len=45, points=500):
  """
  A ray remote function which collects a series of memory-dynamics pairs
  and returns a dataset.
  """
  policy = deepcopy(policy)
  torch.set_num_threads(1)
  with torch.no_grad():
    done = True

    damps  = []
    masses = []
    ipos  = []

    latent = []
    ts     = []

    last = time.time()
    while len(latent) < points:
      x, d, m, q = collect_point(policy, max_traj_len)
      damps   += [d]
      masses  += [m]
      ipos    += [q]
      latent  += [x]
    return damps, masses, ipos, latent
  
def concat(datalist):
  """
  Concatenates several datasets into one larger
  dataset.
  """
  damps    = []
  masses   = []
  ipos    = []
  latents  = []
  for l in datalist:
    damp, mass, quat, latent = l
    damps  += damp
    masses += mass
    ipos   += quat

    latents += latent
  damps   = torch.tensor(damps).float()
  masses  = torch.tensor(masses).float()
  ipos    = torch.tensor(ipos).float()
  latents = torch.tensor(latents).float()
  return damps, masses, ipos, latents

def run_experiment(args):
  """
  The entry point for the dynamics extraction algorithm.
  """
  from util_old import create_logger

  locale.setlocale(locale.LC_ALL, '')

  policy = torch.load(args.policy)

  env_fn = env_factory(args.exp_conf_path)

  layers = [int(x) for x in args.layers.split(',')]

  env = env_fn()
  policy.init_hidden_state()
  policy(torch.tensor(env.reset()).float())
  latent_dim = get_hiddens(policy).shape[0]

  models = []
  opts   = []
  for fn in [env.get_damping, env.get_mass, env.get_ipos]:
    output_dim = fn().shape[0]
    model      = Model(latent_dim, output_dim, layers=layers)

    models += [model]
    opts   += [optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)]

    model.policy_path = args.policy

  logger = create_logger(args)

  best_loss = None
  actor_dir = os.path.split(args.policy)[0]
  create_new = True
  if os.path.exists(os.path.join(logger.dir, 'test_latents.pt')):
    x      = torch.load(os.path.join(logger.dir, 'train_latents.pt'))
    test_x = torch.load(os.path.join(logger.dir, 'test_latents.pt'))

    damps  = torch.load(os.path.join(logger.dir, 'train_damps.pt'))
    test_damps   = torch.load(os.path.join(logger.dir, 'test_damps.pt'))

    masses = torch.load(os.path.join(logger.dir, 'train_masses.pt'))
    test_masses  = torch.load(os.path.join(logger.dir, 'test_masses.pt'))

    ipos  = torch.load(os.path.join(logger.dir, 'train_ipos.pt'))
    test_ipos   = torch.load(os.path.join(logger.dir, 'test_ipos.pt'))

    if args.points > len(x) + len(test_x):
      create_new = True
    else:
      create_new = False
  
  if create_new:
    if not ray.is_initialized():
      ray.init(num_cpus=args.workers)

    print("Collecting {:4d} timesteps of data.".format(args.points))
    points_per_worker = max(args.points // args.workers, 1)
    start = time.time()

    damps, masses, ipos, x = concat(ray.get([collect_data.remote(policy, points=points_per_worker) for _ in range(args.workers)]))

    split = int(0.8 * len(x))

    test_x = x[split:]
    x = x[:split]

    test_damps = damps[split:]
    damps = damps[:split]

    test_masses = masses[split:]
    masses = masses[:split]

    test_ipos = ipos[split:]
    ipos = ipos[:split]

    print("{:3.2f} to collect {} timesteps.  Training set is {}, test set is {}".format(time.time() - start, len(x)+len(test_x), len(x), len(test_x)))
    torch.save(x, os.path.join(logger.dir, 'train_latents.pt'))
    torch.save(test_x, os.path.join(logger.dir, 'test_latents.pt'))

    torch.save(damps, os.path.join(logger.dir, 'train_damps.pt'))
    torch.save(test_damps, os.path.join(logger.dir, 'test_damps.pt'))

    torch.save(masses, os.path.join(logger.dir, 'train_masses.pt'))
    torch.save(test_masses, os.path.join(logger.dir, 'test_masses.pt'))

    torch.save(ipos, os.path.join(logger.dir, 'train_ipos.pt'))
    torch.save(test_ipos, os.path.join(logger.dir, 'test_ipos.pt'))

  for epoch in range(args.epochs):

    random_indices = SubsetRandomSampler(range(len(x)-1))
    sampler = BatchSampler(random_indices, args.batch_size, drop_last=False)

    for j, batch_idx in enumerate(sampler):
      batch_x = x[batch_idx]#.float()
      batch = [damps[batch_idx], masses[batch_idx], ipos[batch_idx]]

      losses = []
      for model, batch_y, opt in zip(models, batch, opts):
        loss = 0.5 * (batch_y - model(batch_x)).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())

      print("Epoch {:3d} batch {:4d}/{:4d}      ".format(epoch, j, len(sampler)-1), end='\r')

    train_y = [damps, masses, ipos]
    test_y  = [test_damps, test_masses, test_ipos]
    order   = ['damping', 'mass', 'com']

    with torch.no_grad():
      print("\nEpoch {:3d} losses:".format(epoch))
      for model, y_tr, y_te, name in zip(models, train_y, test_y, order):
        loss_total = 0.5 * (y_tr - model(x)).pow(2).mean().item()

        preds = model(test_x)
        test_loss  = 0.5 * (y_te - preds).pow(2).mean().item()
        pce = torch.mean(torch.abs((y_te - preds) / (y_te + 1e-5)))
        err = torch.mean(torch.abs((y_te - preds)))
        
        logger.add_scalar(logger.arg_hash + '/' + name + '_loss', test_loss, epoch)
        logger.add_scalar(logger.arg_hash + '/' + name + '_percenterr', pce, epoch)
        logger.add_scalar(logger.arg_hash + '/' + name + '_abserr', err, epoch)
        model.dyn_parameter = name
        torch.save(model, os.path.join(logger.dir, name + '_extractor.pt'))
        print("\t{:16s}: train loss {:7.6f} test loss {:7.6f}, err {:5.4f}, percent err {:3.2f}".format(name, loss_total, test_loss, err, pce))

