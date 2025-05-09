import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Union

# pylint: disable = abstract-method, arguments-differ

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


class Classifier(pl.LightningModule):
    """ Implements the linear classification head used for evaluation of the embeddings. 
    It is used in conjunction with a dataset containing the computed embeddings corresponding 
    to the current encoder considered.
    """

    def __init__(self, config: dict, weight_decay: float = 0):
        super().__init__()
        self.config = config

        if config['pretraining']['dataset']['name'] == "Cifar10":
            self.output_dim = 10
        elif config['pretraining']['dataset']['name'] == "Cifar100":
            self.output_dim = 100
        else:
            raise NotImplementedError(
                f"dataset {config['pretraining']['dataset']} not implemented!")

        self.output = nn.Linear(
            config["pretraining"]["model_dimension"]["embedding_dimension"],
            self.output_dim)
        self.loss = nn.CrossEntropyLoss()

        self.weight_decay = weight_decay

        self.save_hyperparameters(config) #do we need this?


    def forward(self, x) -> torch.tensor:
        """returns the classification results as logits """
        x = self.output(x)
        return F.log_softmax(x, dim=0)

    def training_step(self, batch, batch_idx):
        """performs one training step"""
        x, y = batch
        logits = self.output(x)
        loss = self.loss(logits, y)
        #self.log("testing--training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """performs one validation step"""
        x, y = batch
        logits = self.output(x)
        loss = self.loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (float(len(y)))
        output = {
            "val_loss": loss,
            "val_accuracy": accuracy,
        }
        self.log("validation--val_loss", loss)
        self.log("validation--val_accuracy", accuracy)
        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.output(x)
        loss = self.loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (float(len(y)))
        output = {
            "test_loss": loss,
            "test_accuracy": accuracy,
        }
        self.log("testing--test_loss", loss)
        self.log("testing--test_accuracy", accuracy)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config["testing"]["optimizer"]["lr"],
            weight_decay=self.weight_decay)


#Classes and functions for computation of model score for CIFAR10/100

class ModelScoreDataset(Dataset):
    """
        Dataset class for the evaluation of a model using a linear classifier.
        The constructor takes as input the precomputed embeddings and labels. 
    """

    def __init__(self,
                 embeddings: torch.Tensor,
                 labels: torch.Tensor) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        return self.embeddings[index], self.labels[index]


def compute_model_score(config: dict, 
                        encoder: pl.LightningDataModule,
                        seed: int):
    """ Computation of a final score for the encoder. 

    Args:
        config: configuration file for chosen model; See config.yaml for an example.
        encoder: model to compute score of
        seed: random seed for reproducable results

    Returns:
        In a train/val split manner using the train data, several linear classifiers
        with different weight_decay rates (see config file) are trained and
        validated on the embeddings obtained using the encoder. The weight_decay rate
        with the best validation accuracy is chosen and a linear classifier is trained
        with this weight_decay rate using the whole training set. The output of the function
        (the "model score") is then the accuracy on the test set (not used during training.)
    """
   
    #device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #initialize train and test dataset

    try:
        dataset_name = config['testing']['dataset']['name']
    except KeyError:
        raise NameError("No dataset name in testing_config")
    if dataset_name == 'Cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
                    train=True, download=True, root='./data/CIFAR10',
                    transform=Compose([ToTensor(), Cifar10_normalize]))

        test_dataset = torchvision.datasets.CIFAR10(
                    train=False, download=True, root='./data/CIFAR10',
                    transform=Compose([ToTensor(), Cifar10_normalize]))
    
    elif dataset_name == 'Cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
                    train=True, download=True, root='./data/CIFAR100',
                    transform=Compose([ToTensor(), Cifar100_normalize]))

        test_dataset = torchvision.datasets.CIFAR100(
                    train=False, download=True, root='./data/CIFAR100',
                    transform=Compose([ToTensor(), Cifar100_normalize]))
    
    else:
        raise NotImplementedError(
            f"The dataset {dataset_name} is not supported."
        )

    #computation of embeddings of images in train_dataset unsing encoder in eval mode
    #and saving them together with the corresponding label
    encoder.eval()

    intermediate_loader = DataLoader(dataset=train_dataset, 
                                        batch_size=1024)
    
    stored_embedding_batches = []
    stored_label_batches = []

    for images, labels in intermediate_loader:
        with torch.no_grad():
            stored_embedding_batches.append(encoder(images.to(device)))
        stored_label_batches.append(labels)
    
    train_embeddings = torch.cat(stored_embedding_batches, dim=0)
    train_labels = torch.cat(stored_label_batches, dim=0)

    #create train_labels list (needed for train_test_split further below)
    train_labels_list = train_labels.tolist()

    #create training ModelScoreDataset
    full_training_dataset = ModelScoreDataset(embeddings=train_embeddings, labels=train_labels)

    #create stratified train,val split of full_training_dataset for hyperparameter search (weight_decay)
    #for linear classifier
    val_size = config['testing']['dataset']['val_percentage']

    train_split_indices, val_split_indices, _, _ = train_test_split(range(len(full_training_dataset)),
                                                                    train_labels_list,
                                                                    stratify=train_labels_list,
                                                                    test_size=val_size,
                                                                    random_state=seed)
    
    train_split_dataset = Subset(full_training_dataset, train_split_indices)
    val_split_dataset = Subset(full_training_dataset, val_split_indices)

    #create train_split and val_split dataloader
    train_split_dataloader = DataLoader(
            dataset=train_split_dataset,
            batch_size=config['testing']['batch_size'],
            shuffle=True,
            num_workers=config['testing']['num_workers_loader'])

    val_split_dataloader = DataLoader(
            dataset=val_split_dataset,
            batch_size=config['testing']['batch_size'],
            shuffle=False,
            num_workers=config['testing']['num_workers_loader'])

    #training of linear classifiers using different weight_decay rates to determine optimal rate
    weight_decay_rates = torch.logspace(
                config['testing']['param_tuning']['log_min'],
                config['testing']['param_tuning']['log_max'],
                config['testing']['param_tuning']['n_steps']
                )
    accuracies = []

    for wd in weight_decay_rates:
        testing_module = Classifier(config, weight_decay=wd)
        trainer = pl.Trainer(
            enable_checkpointing=False,
            gpus=torch.cuda.device_count(),
            max_epochs=config["testing"]["max_epochs"],
            #fast_dev_run=True,
            #overfit_batches=1
        )
        trainer.fit(testing_module, train_dataloaders=train_split_dataloader)
        
        #compute validation accuracy 
        trainer_val = pl.Trainer(gpus=torch.cuda.device_count())
        val_output = trainer_val.validate(model=testing_module, dataloaders=val_split_dataloader)
        accuracies.append(val_output[0]['validation--val_accuracy'])

    #choose weight_decay_rate with best validation accuracy
    wd = weight_decay_rates[torch.argmax(torch.tensor(accuracies)).item()]

    #computation of embeddings of images in test_dataset unsing encoder in eval mode
    #and saving them together with the corresponding label
    intermediate_loader = DataLoader(dataset=test_dataset, 
                                        batch_size=1024)
    
    stored_embedding_batches = []
    stored_label_batches = []

    for images, labels in intermediate_loader:
        with torch.no_grad():
            stored_embedding_batches.append(encoder(images.to(device)))
        stored_label_batches.append(labels)
    
    test_embeddings = torch.cat(stored_embedding_batches, dim=0)
    test_labels = torch.cat(stored_label_batches, dim=0)

    #create testing ModelScoreDataset
    final_test_dataset = ModelScoreDataset(embeddings=test_embeddings, labels=test_labels)

    #create full_train, test dataloader
    full_train_loader = DataLoader(
            dataset=full_training_dataset,
            batch_size=config['testing']['batch_size'],
            shuffle=True,
            num_workers=config['testing']['num_workers_loader'])

    test_loader = DataLoader(
            dataset=final_test_dataset,
            batch_size=config['testing']['batch_size'],
            shuffle=False,
            num_workers=config['testing']['num_workers_loader'])
    
    #train linear classifier with determined weight_decay on full_training_dataset and compute
    #its accuracy on final_test_dataset as the model score
    testing_module = Classifier(config, weight_decay=wd)
    trainer = pl.Trainer(
        enable_checkpointing=False,
        gpus=torch.cuda.device_count(),
        max_epochs=config["testing"]["max_epochs"],
        #log_every_n_steps=1, #REMOVE AFTERWARDS -> sets it to 50
        #fast_dev_run=True,  # runs 1 batch, Debugging
        #overfit_batches=1,  # only uses fixed amount of samples, Debugging
    )

    trainer.fit(testing_module,
                train_dataloaders=full_train_loader,
                val_dataloaders=test_loader)

    trainer_test = pl.Trainer(gpus=torch.cuda.device_count())
    test_results = trainer_test.test(model=testing_module, dataloaders=test_loader)

    #set encoder back to training mode in case we train more afterwards
    encoder.train()

    #return obtainend accuracy as final model score
    return test_results[0]['testing--test_accuracy']



    
