# Standard Library
from statistics import mean

# Third Party
import torch
from torch.nn import MSELoss

from tqdm.auto import tqdm, trange



###########
# UTILITIES
###########


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(model, train_set, encoding_dim, **kwargs):
    
    if model.__name__ in ("LINEAR_AE", "LSTM_AE"):
        return model(train_set[-1].shape[-1], encoding_dim, **kwargs)
    elif model.__name__ == ("LSTM_VAE"):
        return model(
                        input_size=train_set[-1].shape[-1],
                        hidden_size=kwargs["h_dim"],
                        latent_size=encoding_dim,
                        device=get_device(),
                    )
    
    elif model.__name__ == "CONV_LSTM_AE":
        if len(train_set[-1].shape) == 3:  # 2D elements
            return model(train_set[-1].shape[-2:], encoding_dim, **kwargs)
        elif len(train_set[-1].shape) == 4:  # 3D elements
            return model(train_set[-1].shape[-3:], encoding_dim, **kwargs)

def train_model(
                    model, 
                    train_set, 
                    verbose, 
                    lr, 
                    epochs, 
                    clip_value, 
                    device=None,pbar_conf=None
                ):
    if device is None:
        device = get_device()
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(reduction='mean')

    mean_losses = []


    pbar_id = pbar_conf['id'] if pbar_conf is not None else 0
    text = "# trng {0}".format(pbar_id)   
    trange_epoch =  trange(
                            epochs, 
                            desc=text, 
                            disable= False,
                            leave=False,
                            lock_args=None ,
                            position=2*pbar_id
                        )

    # make batches of trajs of same length
    traj_set_batches = {}
    for traj in train_set:
        if traj.shape[0] not in traj_set_batches.keys():
            traj_set_batches[traj.shape[0]] = []
        traj_set_batches[traj.shape[0]].append(traj)

    for key in traj_set_batches.keys():
        traj_set_batches[key] = torch.stack(traj_set_batches[key]).to(device)
        # print(key, traj_set_batches[key].shape)

    print("n_traj_batches:",len(traj_set_batches.keys()))

    for epoch in trange_epoch:

        model.train()

        # # Reduces learning rate every 50 epochs
        # if not epoch % 50:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr * (0.993 ** epoch)

        losses = []
        # for x in train_set:
        # x = x.to(device)
        optimizer.zero_grad()

        loss = 0
        for key in traj_set_batches.keys():
            # Forward pass
            x_prime = model(traj_set_batches[key])
            loss += criterion(x_prime, traj_set_batches[key])
        
        loss = loss/len(traj_set_batches.keys())                
        # Backward pass
        loss.backward()


        # Gradient clipping on norm
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        losses.append(loss.item())

        mean_loss = mean(losses)
        mean_losses.append(mean_loss)

        if verbose:
            print(f"Epoch: {epoch}, Loss: {mean_loss}")

        trange_epoch.set_description_str(
                                        "ep:"+str(round(epoch,2))+" ls:"+str(round(mean_loss,2)),
                                        refresh=True
                                        )

    return mean_losses

def train_vae_model(
                    model, 
                    train_set,
                    verbose,
                    lr,
                    epochs,
                    device=None,
                    pbar_conf = None
                    ):

    if device is None:
        device = get_device()
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trng_loss = []

    pbar_id = pbar_conf['id'] if pbar_conf is not None else 0
    text = "# trng {0}".format(pbar_id)   
    trange_epoch =  trange(
                            epochs, 
                            desc=text, 
                            disable= False,
                            leave=False,
                            lock_args=None ,
                            position=2*pbar_id
                        )
    

    ## training
    count = 0
    for epoch in trange_epoch:

        model.train()
        optimizer.zero_grad()

        mloss, recon_x,z, info = model(train_set)

        # Backward and optimize
        optimizer.zero_grad()
        mloss.mean().backward()

        optimizer.step()
            
        # writer.add_scalar("train_loss", float(mloss.mean()), epoch)
        mean_loss = float(mloss.mean())
        trng_loss.append(mean_loss)

        if verbose:
            print(f"Epoch: {epoch}, Loss: {mean_loss}")
        trange_epoch.set_description_str(
                                        "ep:"+str(round(epoch,2))+" ls:"+str(round(mean_loss,2)),
                                        refresh=True
                                        )



    return trng_loss

def get_encodings(model, train_set, device=None):

    zs = []
    z_means = []
    z_stds = []
    if device is None:
        device = get_device()
    model.eval()

    for x in train_set:
        z = model.encoder(x.to(device))

        zs.append(z.detach().cpu().numpy())
        z_means.append(z.detach().cpu().numpy())
        z_stds.append(0)
    
    return {'samples':zs,'means':z_means,'stds':z_stds}

def get_vae_encodings(model, train_set, device=None):
    zs = []
    z_means = []
    z_stds = []

    model.eval()
    for x in train_set:
        x = x.unsqueeze(0)
        _,x_hat,z_stats,info = model(x)

        z = z_stats['sample']
        zs.append(z[0,0,:].detach().cpu().numpy())
        z_means.append(z_stats['mean'].detach().cpu().numpy()[0])
        z_stds.append(z_stats['std'].detach().cpu().numpy()[0])
    return {'samples':zs,'means':z_means,'stds':z_stds}
######
# MAIN
######


def quick_train(
    model,
    train_set,
    encoding_dim,
    verbose=False,
    lr=1e-3,
    epochs=50,
    clip_value=1,
    denoise=False,
    device=None,
    seed=0,
    pbar_conf = {},
    **kwargs,
):
    


    torch.manual_seed(seed)

    load_model_frm = None    
    if 'load_model_frm' in kwargs.keys():
        load_model_frm = kwargs['load_model_frm']
        kwargs.pop('load_model_frm')

    is_vae = True if model.__name__ == "LSTM_VAE" else False
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)

    if device is None:
        device = get_device()
        print("model initialised in device:", device)

    if load_model_frm is not None:
        print("loading model from:",load_model_frm)

        # print(torch.load(load_model_frm))
        model.encoder.load_state_dict(torch.load(load_model_frm).state_dict())
        model.decoder.load_state_dict(torch.load(load_model_frm.replace('enc','dec')).state_dict())
    
    if is_vae:
        if isinstance(train_set, list):
            train_set = torch.stack(train_set).to(device)

        losses = train_vae_model(
                                    model, 
                                    train_set, 
                                    verbose, 
                                    lr, 
                                    epochs, 
                                    device, 
                                    pbar_conf
                                )

        z_stats = get_vae_encodings(model, train_set, device)

    else:
        losses = train_model(
                                model, 
                                train_set, 
                                verbose, 
                                lr, 
                                epochs, 
                                clip_value, 
                                device, 
                                pbar_conf
                            )
        z_stats = get_encodings(model, train_set, device)

    return model, z_stats, losses