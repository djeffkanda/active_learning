import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, Subset
from typing import Union



class MaskSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)


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

        self.current_train_set = Subset(train_dataset, train_labeled_index)

        train_sampler, val_sampler = self.train_validation_split(len(self.current_train_set), validation, seed)
        self.train_loader = DataLoader(self.current_train_set, batch_size, sampler=train_sampler, **kwargs)
        self.validation_loader = DataLoader(self.current_train_set, batch_size, sampler=val_sampler, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)

    def get_unlabeled_data(self):
        return Subset(self.train_set, (self.train_unlabeled_mask == 0).nonzero())

    def update_train_set(self,):
        # TODO implement this function according to the discussion tomorrow
        return

    @staticmethod
    def train_validation_split(num_samples, validation_ratio, seed=0):
        """
        Returns two torch Samplers, one for training and the other for validation.
        Both samplers are used with the training dataset (see __init__).

        Args:
            num_samples: number of samples to split between train and val set
            validation_ratio: proportion of the train dataset used for validation set
            seed: random seed for splitting train and validation set
        Returns:
            train and validation samplers
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
