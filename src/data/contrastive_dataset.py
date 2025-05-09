""" The implementation of a contrastive dataset.
"""
from typing import Union

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Normalize, Compose
from torch.utils.data import DataLoader

from src.data.transformations import get_transform  # pylint: disable = import-error

#DATASET NORMALIZATION (values computed in file .... using only training data)
Cifar10_values = {
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2470, 0.2435, 0.2616)
}
Cifar100_values = {
    'mean': (0.5071, 0.4865, 0.4409),
    'std': (0.2673, 0.2564, 0.2762)
}

Cifar10_normalize = Normalize(Cifar10_values['mean'], Cifar10_values['std'])
Cifar100_normalize = Normalize(Cifar100_values['mean'], Cifar100_values['std'])


class ContrastiveDataset(Dataset):
    """ This class "augments" a dataset to a contrastive one.

    Pictures from a base-dataset are augmented with the specified transforms and
    arranged in positive pairs as needed for a pairwise positive loss
    (cl_loss_pairwise_positive, after feature extraction).
    """

    def __init__(self,
                 dataset_config: dict,
                 dataset: Union[None, Dataset] = None) -> None:
        """ 
        Args:
            dataset_config: Only the config_file part concerned with the dataset.
                See the default config in config/config.yaml for an example.
            dataset: None if we want to use dataset with name specified in dataset_config; actual Dataset if we want to use this 
                dataset instead.
        """
        super().__init__()
        if isinstance(dataset, Dataset):
            self.base_dataset = dataset

        if dataset is None:
            try:
                dataset_name = dataset_config['name']
            except KeyError:
                raise NameError("No dataset name in dataset_config")
            if dataset_name == "Cifar10":
                self.base_dataset = torchvision.datasets.CIFAR10(
                    train=True,
                    download=True,
                    root=
                    './data/CIFAR10'  #need to run script in code folder for this to work
                )
                self.transform = Compose([
                    get_transform(dataset_config["transforms"]),
                    Cifar10_normalize
                ])
            elif dataset_name == "Cifar100":
                self.base_dataset = torchvision.datasets.CIFAR100(
                    train=True,
                    download=True,
                    root=
                    './data/CIFAR100'  #need to run script in code folder for this to work
                )
                self.transform = Compose([
                    get_transform(dataset_config["transforms"]),
                    Cifar100_normalize
                ])
            else:
                raise NotImplementedError(
                    f"The dataset {dataset_name} is not implemented.")
        else:
            raise TypeError(
                f"Expected torch.data.Dataset or None, but got {type(dataset)}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index) -> torch.Tensor:
        """Returns:
            Given an index of an image in the dataset, it returns two distinct augmentations of the same image 
            (the augmentation is specified in dataset_config) in a torch tensor
        """
        image1, _ = self.base_dataset[index]
        image2, _ = self.base_dataset[index]

        return torch.stack((self.transform(image1), self.transform(image2)), dim=0)
        #return torch.stack((torchvision.transforms.ToTensor()(image1), torchvision.transforms.ToTensor()(image2)), dim=0)

#Collate function for dataloader (needed to generate correct batches)
def collate_fn(batch):
    return torch.cat(batch, dim=0)


def get_contrastive_trainloader(pretraining_config: dict,
                                dataset: Union[None, Dataset] = None):
    """"
    Args:
        pretraining_config: Only config_file part concerned with pretraining. See the default config in config/config.yaml
        for an example
        dataset: None if we want to use dataset with name specified in dataset_config; actual Dataset if we want to use this 
                dataset instead.
    
    Returns:
        Torch.utils.data.Dataloader with parameters specified in pretraining_config. 
        The batch output of this dataloader is a tensor [x_0,x_1, ..., x_(2n-2), x_(2n-1)], where n=batch_size, and
        x_2k and x_2k+1 are different augmentations of the same image
    """

    #TODO: num_workers = number of cpu cores; os.cpu_count()

    train_dataset = ContrastiveDataset(pretraining_config['dataset'],
                                       dataset=dataset)
    return DataLoader(dataset=train_dataset,
                      batch_size=pretraining_config['batch_size'],
                      shuffle=True,
                      num_workers=pretraining_config['num_workers_loader'],
                      collate_fn=collate_fn)
