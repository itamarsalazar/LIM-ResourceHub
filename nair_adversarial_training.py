import os
import sys

sys.path.insert(0, '/nfs/privileged/isalazar/projects/ultrasound-image-formation/')
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

from src.models.unet.unet_model import UNet
from src.models.dataloaders.cystDataset import CystDatasetEMBC
from utils import create_run_description, attack_model_during_training
from torch.optim.lr_scheduler import CyclicLR


def splitDataloader(full_dataset, batch_size, train_val_split, indices_dir, previous_indices=False, save_indices=False):
    if previous_indices:
        # Load the saved indices from the provided directory
        training_info = np.load(os.path.join(indices_dir, 'training_info.npy'), allow_pickle='TRUE').item()
        training_info['train_indices'].sort()
        training_info['val_indices'].sort()

        train_indices = np.asarray(training_info['train_indices'])
        val_indices = np.asarray(training_info['val_indices'])

        print('Previous indices correctly loaded')

    else:
        # Split the dataset randomly into training and validation sets
        train_size = int(train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices

        if save_indices:
            # Save the indices for future use
            os.makedirs(os.path.join(indices_dir), exist_ok=True)
            with open(os.path.join(indices_dir, 'train_indices.pkl'), 'wb') as f:
                pickle.dump(train_indices, f)
            with open(os.path.join(indices_dir, 'val_indices.pkl'), 'wb') as f:
                pickle.dump(val_indices, f)
            print('Random split. Indices correctly saved')

    # Create subsets for training and validation sets based on the provided indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print("train dataset: %d elements, val dataset: %d elements" % (len(train_dataset), len(val_dataset)))

    # Create the dataloaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size[1], shuffle=False)

    return train_loader, val_loader


def load_model(model_dir=None, epoch=None, device=None):
    # Instantiate the model architecture
    model = UNet(n_channels=2, n_classes=1).to(device)

    if isinstance(epoch, int) and epoch == -1:
        prev_history = {"train_loss": [], "val_loss": []}
        print("Model initialized with random weights")
    else:
        if isinstance(epoch, int):
            model_filename = os.path.join(model_dir, 'model_%.4d.t7' % epoch)
            history_filename = os.path.join(model_dir, 'history_%.4d.pkl' % epoch)
            print(f"Loading model {epoch}...")
        elif isinstance(epoch, str):
            model_filename = os.path.join(model_dir, 'model_%s.t7' % epoch)
            history_filename = os.path.join(model_dir, 'history_%s.pkl' % epoch)
            print(f"Loading model {epoch}...")
        else:
            raise ValueError("Invalid epoch type. Should be either int or str.")

        # If weights path is provided, load the weights from the saved model
        if os.path.isfile(model_filename) and os.path.isfile(history_filename):
            checkpoint = torch.load(model_filename, map_location=device)
            model.load_state_dict(checkpoint)
            with open(history_filename, 'rb') as f:
                prev_history = pickle.load(f)
            print(f"Model {epoch} loaded.")
        else:
            raise ValueError(f"No weights or history found at {model_dir}.")

    return model, prev_history


def adversarial_train(model, epsilon, prev_hist, train_loader, val_loader, init_epoch, last_epoch, opt, clr,
                      loss_fun, loss_adv, model_dir, device):
    history = {"train_loss": [*prev_hist['train_loss']],
               "val_loss": [*prev_hist['val_loss']]}

    for epoch in range(init_epoch, last_epoch):
        train_loss_sum = 0
        train_n = 0

        model.train()
        for batch_idx, samples in enumerate(train_loader):
            channel_data, targets = samples['input'], samples['target']
            channel_data, targets = channel_data.to(device), targets.to(device)

            ###############################################################################
            # Generate the adversarial input
            if epsilon == 0.0:
                print("|", end='\n')
                delta = torch.zeros_like(channel_data)
            else:
                delta = None
                delta, _ = attack_model_during_training(model,
                                                        case='nair',
                                                        X=channel_data,
                                                        targets=targets,
                                                        epsilon=epsilon,
                                                        alpha=2 * epsilon / 20,
                                                        attack_iters=20,
                                                        lower_limit=-1,
                                                        upper_limit=1,
                                                        p='inf',
                                                        device=device,
                                                        loss_fn=loss_adv)
            channel_data_perturbed = channel_data + delta
            channel_data_perturbed = channel_data_perturbed / channel_data_perturbed.abs().max()
            ###############################################################################

            model.train()
            outputs_beam = model(channel_data_perturbed)

            loss = loss_fun(outputs_beam, targets)

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            clr.step()
            current_lr = opt.param_groups[0]['lr']

            train_loss_sum += loss.item() * targets.size(0)
            train_n += targets.size(0)
            history["train_loss"].append(loss.item())

            if batch_idx % 50 == 0:
                print('\nEpoch %d: batch %d of %d |train loss: %.5f | lr: %s ' %
                      (epoch, batch_idx, len(train_loader) - 1, train_loss_sum / train_n, current_lr), end="")
                with open(os.path.join(model_dir, 'batch.txt'), 'w') as f:
                    # Redirect stdout to the file
                    sys.stdout = f
                    # Print some messages
                    print('\nEpoch %d: batch %d of %d |train loss: %.5f | lr: %s ' %
                          (epoch, batch_idx, len(train_loader) - 1, train_loss_sum / train_n, current_lr))
                    # Reset stdout back to the console
                    sys.stdout = sys.__stdout__
            print('.', end="")
        train_loss = train_loss_sum / train_n
        val_loss = test(model, val_loader, loss_fun)
        history["val_loss"].append(val_loss)
        # # Open a file for writing
        with open(os.path.join(model_dir, 'epoch.txt'), 'w') as f:
            # Redirect stdout to the file
            sys.stdout = f
            # Print some messages
            print('\n\tFinished Epoch:%d |Train Loss: %.5f |Val Loss (' % (epoch, train_loss), end="")
            print('%s:%.5f)' % ('Val Loss', val_loss))
            # Reset stdout back to the console
            sys.stdout = sys.__stdout__
        print('\n\tFinished Epoch:%d |Train Loss: %.5f |Val Loss (' % (epoch, train_loss), end="")
        print('%s:%.5f)' % ('Val Loss', val_loss))
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()

        save_model_and_history('intermediate', state_dict, history, model_dir)
        if (epoch + 1) % 10 == 0:
            save_model_and_history('%.4d' % int(epoch), state_dict, history, model_dir)

    save_model_and_history('last', state_dict, history, model_dir)
    return model, history


def save_model_and_history(label, state_dict, history, model_dir):
    torch.save(state_dict, os.path.join(model_dir, 'model_%s.t7') % label)
    history_filename = os.path.join(model_dir, 'history_%s.pkl' % label)
    with open(history_filename, 'wb') as f:
        pickle.dump(history, f)


def test(model, valid_loader, loss_fun):
    device = next(model.parameters()).device.type
    test_loss_sum = 0
    test_n = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, samples in enumerate(valid_loader):
            input, targets = samples['input'], samples['target']
            input, targets = input.to(device), targets.to(device)
            outputs_beam = model(input)
            loss = loss_fun(outputs_beam, targets)
            test_loss_sum += loss.item() * targets.size(0)
            test_n += targets.size(0)
        test_loss = (test_loss_sum / test_n)
    return test_loss


def create_dir(my_dir):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
        print(f"Just created: {my_dir}")
    else:
        print(f"Already exists: {my_dir}")


if __name__ == '__main__':
    # this_dir = ('C:/Users/u_imagenes/PycharmProjects/ultrasound-image-formation/exploration/'
    #             'Journal2023/')
    # data_dir = ('C:/Users/u_imagenes/PycharmProjects/ultrasound-image-formation/exploration/'
    #             'Journal2023/sample_data/TUFFC/')
    this_dir = '/nfs/privileged/isalazar/projects/ultrasound-image-formation/exploration/Journal2023/'
    data_dir = '/nfs/privileged/isalazar/datasets/simulatedCystDataset/TUFFC'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########################################################
    load_epoch = -1  # set to -1 to train from scratch
    last_epoch = 100
    lr_base = 1e-6
    lr_max = 1e-3
    BATCH_SIZE = 12  # 16
    eps = 0.008
    ########################################################
    print('Device is: %s' % device)
    save_model_dir = os.path.join(this_dir, "models", f'nair_adv_eps{str(eps)}')
    create_dir(save_model_dir)
    description = f"""
    Adversarial training: 
    - Adversarial loss: Same as training
    - epsilon: 0.008, in kireev2022a, they found 1/255 to be the "sweet spot". This is approx 0.008 in our case.
    """
    create_run_description(save_model_dir, description)

    # Create dataloaders
    full_dataset = CystDatasetEMBC(data_dir=data_dir,
                                   input_subdir='input_id',
                                   target_subdir='target_enh')

    train_loader, val_loader = splitDataloader(full_dataset,
                                               batch_size=(BATCH_SIZE, BATCH_SIZE),
                                               train_val_split=None,
                                               indices_dir=os.path.join(this_dir, "models"),
                                               previous_indices=True)   # <---------------------
    # train_loader, val_loader = splitDataloader(full_dataset,
    #                                            batch_size=(1, 1),
    #                                            train_val_split=0.8,
    #                                            indices_dir=os.path.join(this_dir, "models"),
    #                                            previous_indices=False)

    model, prev_history = load_model(model_dir=save_model_dir, epoch=load_epoch, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr_base)
    clr = CyclicLR(opt, base_lr=lr_base, max_lr=lr_max, step_size_up=2 * len(train_loader), mode='triangular2', cycle_momentum=False)
    model, history = adversarial_train(model,
                                       epsilon=eps,
                                       prev_hist=prev_history,
                                       train_loader=train_loader,
                                       val_loader=val_loader,
                                       init_epoch=load_epoch + 1,
                                       last_epoch=last_epoch,
                                       opt=opt,
                                       clr=clr,
                                       loss_fun=nn.L1Loss(),
                                       loss_adv=nn.MSELoss(),
                                       model_dir=save_model_dir,
                                       device=device)
