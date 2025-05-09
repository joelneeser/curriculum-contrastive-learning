import torch
from torch._C import dtype
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Union

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



#FILE NOT NEEDED ANYMORE -> DELETE?


class ClassificationDataset(Dataset):
    """This class provides a simple dataloader for the classification task"""

    def __init__(self,
                 dataset_config: dict,
                 encoder: pl.LightningModule,
                 train: bool = True,
                 dataset: Union[None, Dataset] = None) -> None:
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
                    train=train, download=True, root='./data/CIFAR10',
                    transform=Compose([ToTensor(), Cifar10_normalize]))
                self.num_classes = 10

            elif dataset_name == "Cifar100":
                self.base_dataset = torchvision.datasets.CIFAR100(
                    train=train, download=True, root='./data/CIFAR100',
                    transform=Compose([ToTensor(), Cifar100_normalize]))
                self.num_classes = 100

            else:
                raise NotImplementedError(
                    f"The dataset {dataset_name} is not implemented.")
        else:
            raise TypeError(
                f"Expected torch.data.Dataset or None, but got {type(dataset)}")

        self.encoder = encoder
        self.encoder.eval()

        #computation of all embeddings + saving them together with the corresponding label
        intermediate_loader = DataLoader(dataset=self.base_dataset, 
                                        batch_size=1024)
        
        stored_embedding_batches = []
        stored_label_batches = []

        for images, labels in intermediate_loader:
            with torch.no_grad():
                stored_embedding_batches.append(self.encoder(images))
            stored_label_batches.append(labels)
        
        self.embeddings = torch.cat(stored_embedding_batches, dim=0)
        self.labels = torch.cat(stored_label_batches, dim=0)

        # set encoder back to training mode in case we train more afterwards
        self.encoder.train()                   

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index) -> torch.Tensor:
        return self.embeddings[index], self.labels[index]


def get_classification_dataloader_split(testing_config: dict,
                    encoder: pl.LightningModule,
                    dataset: Union[None, Dataset] = None,
                    shuffle: bool = True):
    """
    Args:
        testing_config: part of config concerned with testing
        encoder: the encoder module used to generate the embeddings
        dataset: None if we want to use dataset with name specified in dataset_config; actual Dataset if we want to use this 
                dataset instead.
        shuffle: whether to shuffle the datasets or not

    Returns: tuple: (torch.utils.data.Dataloader, torch.utils.data.Dataloader)
        where the first one is the train dataloader and the second one the validation dataloader
    """

    full_dataset = ClassificationDataset(testing_config['dataset'], encoder, train=True, dataset=dataset)
    val_size = int(len(full_dataset) * testing_config['dataset']['val_percentage'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=testing_config['batch_size'],
            shuffle=shuffle,
            num_workers=testing_config['num_workers_loader'])

    val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=testing_config['batch_size'],
            shuffle=shuffle,
            num_workers=testing_config['num_workers_loader'])

    return train_dataloader, val_dataloader
    

def get_classification_trainloader(testing_config: dict,
                    encoder: pl.LightningModule,
                    train: bool,
                    dataset: Union[None, Dataset] = None,
                    shuffle: bool = True):
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

    train_dataset = ClassificationDataset(testing_config['dataset'], encoder,
                                          train, dataset)
    return DataLoader(
        dataset=train_dataset,
        batch_size=testing_config['batch_size'],
        shuffle=shuffle,
        num_workers=testing_config['num_workers_loader'],
    )
