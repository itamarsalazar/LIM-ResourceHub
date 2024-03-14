import os
from torch.utils.data import Dataset


class CystDatasetEMBC(Dataset):
    def __init__(self, data_dir, input_subdir, target_subdir, transform=None):
        self.data_dir = data_dir
        self.input_subdir = input_subdir
        self.target_subdir = target_subdir
        self.transform = transform

        # get the list of numpy arrays in each subdirectory
        self.input_files = sorted(os.listdir(os.path.join(data_dir, input_subdir)))
        self.target_files = sorted(os.listdir(os.path.join(data_dir, target_subdir)))

        assert len(self.input_files) == len(self.target_files)

        # calculate the length of the dataset
        self.dataset_length = len(self.input_files)

    def __len__(self):
        return self.dataset_length