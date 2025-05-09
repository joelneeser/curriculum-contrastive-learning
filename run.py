""" Main script """

import sys
import os

import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
import yaml
import json

from src.models.pretraining import Pretraining
from src.data.contrastive_dataset import get_contrastive_trainloader
from src.callbacks.loss_modelscore_logger import LossModelscoreLogger

#pylint: disable = anomalous-backslash-in-string

def main(config_name=None):
    """
    Args:
        config_name: name of config file in config folder to use in this run, 
        e.g. config_example.yaml as an example file to be used.
    """
    if config_name is None:
        #no config file indicated
        print('No config file name found. Exiting script.')
        sys.exit()

    else:
        #create config file path
        config_path = "./config/" + config_name

        try: 
            with open(config_path) as file:
                config = yaml.safe_load(file)
        except:
            print('File not found. Exiting script.')
            sys.exit()

    pl.seed_everything(config["seed"], workers=True)

    print('Training configuration:')
    print(config)

    """Training/testing procedure for current config"""

    if config["pretraining"]["training"] is True:
        encoder = Pretraining(config)
        #Preparing contrastive dataloader
        train_loader = get_contrastive_trainloader(config['pretraining'])

        #construct path for logging files for current config file
        path = './experiments/' + config['experiment']

        #init LossModelscoreLogger
        cb = LossModelscoreLogger(config=config)

        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            default_root_dir=path,
            max_epochs=config["pretraining"]["max_epochs"],
            callbacks=[cb],
            #fast_dev_run=True,  # runs 1 batch, Debugging
            #overfit_batches=1,  # only uses fixed amount of samples, Debugging
        )
        trainer.fit(encoder, train_loader)
    
    #get log files from LossModelscoreLogger
    log_dictionary = cb.get_logs()
    print(log_dictionary)

    #save log_dictionary in JSON file
    file_name = './experiment_logs/' + config['experiment'] + '.json'
    
    try:
        os.mkdir('experiment_logs')
    except:
        pass

    with open(file_name, 'w') as fout:
        json.dump(log_dictionary, fout)
        
if __name__ == "__main__":
    try:
        config=sys.argv[1]
    except:
        config=None
    main(config)
