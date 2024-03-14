import os
import sys
sys.path.insert(0, '/nfs/privileged/isalazar/projects/ultrasound-image-formation/')
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset


def splitDataloader(full_dataset, batch_size, train_val_split, indices_dir, previous_indices=False):
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

        # # Save the indices for future use
        # os.makedirs(os.path.join(indices_dir), exist_ok=True)
        # with open(os.path.join(indices_dir, 'train_indices.pkl'), 'wb') as f:
        #     pickle.dump(train_indices, f)
        # with open(os.path.join(indices_dir, 'val_indices.pkl'), 'wb') as f:
        #     pickle.dump(val_indices, f)
        # print('Random split. Indices correctly saved')

    # Create subsets for training and validation sets based on the provided indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print("train dataset: %d elements, val dataset: %d elements" % (len(train_dataset), len(val_dataset)))

    # Create the dataloaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size[1], shuffle=False)

    return train_loader, val_loader

def save_model_and_history(label, state_dict, history, model_dir):
    torch.save(state_dict, os.path.join(model_dir, 'model_%s.t7') % label)
    history_filename = os.path.join(model_dir, 'history_%s.pkl' % label)
    with open(history_filename, 'wb') as f:
        pickle.dump(history, f)


def save_model(label, state_dict, model_dir):
    torch.save(state_dict, os.path.join(model_dir, 'model_%s.t7') % label)


def save_history(label, history, model_dir):
    history_filename = os.path.join(model_dir, 'history_%s.pkl' % label)
    with open(history_filename, 'wb') as f:
        pickle.dump(history, f)


def create_dir(my_dir):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
        print(f"Just created: {my_dir}")
    else:
        print(f"Already exists: {my_dir}")