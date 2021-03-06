import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, Subset


class MaskSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask)[0])

    def __len__(self):
        return len(self.mask)


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class MySubset(Subset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]] + (idx, )


class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 batch_size: int = 1,
                 num_classes: int = None,
                 input_shape: tuple = None,
                 validation: float = 0.1,
                 initial_train_dataset_ratio: float = 0.2,
                 seed: int = 0,
                 **kwargs):
        """
        Args:
            train_dataset: pytorch dataset used for training
            test_dataset: pytorch dataset used for testing
            batch_size: int, size of batches
            num_classes: int number of classes
            input_shape: tuple, shape of the input image
            validation: float, proportion of the train dataset used for the validation set
            seed: int, random seed for splitting train and validation set
            **kwargs: dict with keywords to be used
        """

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.validation = validation
        self.kwargs = kwargs
        self.seed = seed

        torch.manual_seed(seed)
        n = len(train_dataset)
        num_sample = int(n * initial_train_dataset_ratio)
        shuffled_idx = torch.randperm(n).long()

        # Create a mask to track labeling process
        self.train_unlabeled_mask = torch.zeros_like(shuffled_idx)
        self.train_unlabeled_mask[shuffled_idx[:num_sample]] = 1

        train_labeled_index = shuffled_idx[:num_sample]
        self.current_train_set = MySubset(train_dataset, train_labeled_index)

        # Create the loaders
        train_sampler, val_sampler = self.train_validation_split(len(self.current_train_set), self.validation,
                                                                 self.seed)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.current_train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)

    def get_unlabeled_data(self):
        """
        This function return the unlabeled dataset and their corresponding indices.
        :return:
        unlabeled dataset and indices,
        """
        unlabeled_indices = (self.train_unlabeled_mask == 0).nonzero().squeeze()
        unlabeled_data = MySubset(self.train_set, unlabeled_indices)
        return unlabeled_data, unlabeled_indices

    def get_unlabelled_data_loader(self, batch):
        data, idx = self.get_unlabeled_data()
        loader = DataLoader(data, batch, shuffle=False, **self.kwargs)
        return loader, idx

    def get_current_training_set(self):
        return self.current_train_set

    def update_train_set(self, index_to_label):
        """
        This function update the labeled dataset by adding the new data sample to label in the current train set
        :param index_to_label: indices of new data to label
        :return:
        train_loader and validation_loader
        """
        self.train_unlabeled_mask[index_to_label] = 1
        lbl_sample_idx = self.train_unlabeled_mask.nonzero().squeeze()
        self.current_train_set = MySubset(self.train_set, lbl_sample_idx)
        train_sampler, val_sampler = self.train_validation_split(len(self.current_train_set), self.validation,
                                                                 self.seed)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.current_train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        return self.train_loader, self.validation_loader

    @staticmethod
    def train_validation_split(num_samples, validation_ratio, seed=0):
        """
        This function returns two samplers for training and validation data.
        :param num_samples: total number of sample to split
        :param validation_ratio: percentage of validation dataset
        :param seed: random seed to use
        :return:
        """
        torch.manual_seed(seed)
        num_val = int(num_samples * validation_ratio)
        shuffled_idx = torch.randperm(num_samples).long()
        train_idx = shuffled_idx[num_val:]
        val_idx = shuffled_idx[:num_val]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        return train_sampler, val_sampler

    def get_train_set(self):
        return self.train_loader

    def get_validation_set(self):
        return self.validation_loader

    def get_test_set(self):
        return self.test_loader

    def get_classes(self):
        return range(self.num_classes)

    def get_input_shape(self):
        return self.input_shape

    def get_batch_size(self):
        return self.batch_size

    def get_random_sample_from_test_set(self):
        indice = np.random.randint(0, len(self.test_set))
        return self.test_set[indice]

    def getthi(self):
        return  self.train_set
