""" This class implements the pretraining logic of our model, following SimCLR.
"""
import torch
from torch import nn
import pytorch_lightning as pl
from src.networks.ResNet import get_resnet
from src.losses.get_loss import get_loss
from src.models.small_model import SmallModel
from src.models.classification import compute_model_score

class Pretraining(pl.LightningModule):
    """ Contains the main logic of our pretraining pipeline.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        ### Initialize network architecture used.
        # 2 options:
        #   1. We train on 32x32 images (97 MB)
        #   2. We train and test on 224x224 upscaled images (272 MB, average pool in ResNet)
        if "ResNet" in config["encoder_architecture"]["name"]:
            self.encoder = get_resnet(config["encoder_architecture"]["name"])
            self.embedding_dim = 2048

        #for experimenting with training process, gpus, etc.
        if config['encoder_architecture']['name'] == 'SmallModel':
            self.encoder = SmallModel()
            self.embedding_dim = 2048

        self.clloss = get_loss(config["pretraining"]["loss"])

        #Projection_head
        if config["pretraining"]["projection"]["use_projection"]:
            self.projection = nn.Linear(
                self.embedding_dim,
                config["pretraining"]["projection"]["projection_dimension"])
        else:
            self.projection = None
        ###

        self.save_hyperparameters(config)

    def forward(self, x):
        """ Returns the representation/embedding for x."""
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """ Returns the loss for a given batch.

        The batch is assumed to come from a contrastive_dataset, i.e. they
        are already in the required shape for a cl_loss_pairwise_positives.
        """

        embeddings = self.encoder(batch)

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        # Compute loss for the given batch and log loss averaged over all batches in epoch
        loss = self.clloss(embeddings)
        self.log("average_train_loss", loss, on_epoch=True)
        return loss

    def on_epoch_end(self) -> None:
        self.clloss.step()
        
    def configure_optimizers(self):
        optim_config = parse_config_dict(
            self.config["pretraining"]["optimizer"]["config"])
        if self.config["pretraining"]["optimizer"]["name"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **optim_config)
        return optimizer


def parse_config_dict(config: dict) -> dict:
    """ Prepares optimizer_config for use in constructor.

    Args:
        config: Dictionary containing the parameters for the given optimizer.
            E.g., for Adam, {"lr": 0.001, "betas": "0.9,0.999"}.

    Returns:
        The cleaned dict, which can be passed to the optimizer constructor.
    """
    # Parse the dict representations
    cleaned_dict = {}
    for param_name, param_value in config.items():
        if isinstance(param_value, str):
            string_list = param_value.split(",")
            cleaned_param_value = tuple(
                string_to_numerical(str_num) for str_num in string_list)
            cleaned_dict.update({param_name: cleaned_param_value})
        else:
            cleaned_dict.update({param_name: param_value})

    return cleaned_dict


def string_to_numerical(string: str):
    """ Converts numerical int/float strings to numerical values.

    Args:
        string: string of numerical value, e.g., '2.1' or '2'

    Returns:
        Corresponding value as an int if it is an integer, and as a float otherwise.
    """
    try:
        return int(string)
    except ValueError:
        return float(string)
