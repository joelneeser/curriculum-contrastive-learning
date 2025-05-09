import pytorch_lightning as pl
from pytorch_lightning import callbacks

from src.models.classification import compute_model_score


class LossModelscoreLogger(pl.Callback):
    """ Callback for computation of model scores, and logging of model scores and average 
        training loss per epoch.
    """
    def __init__(self, config: dict):
        self.average_training_loss_per_epoch = []
        self.model_scores = []
        self.config = config
    
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #save average_train_loss_epoch for this epoch
        self.average_training_loss_per_epoch.append(trainer.logged_metrics['average_train_loss_epoch'].item())

        #computation of current model score
        if self.config["testing"]["testing"] and isinstance(
                self.config["testing"]["testing_every_n_epochs"], int):

            check_interval = self.config["testing"]["testing_every_n_epochs"]

            if (pl_module.current_epoch+1) % check_interval == 0:
                print(f'Computing model score after {pl_module.current_epoch+1} epochs of training:')
                model_score = compute_model_score(self.config, pl_module, self.config["seed"]) 
            
                #save current model score (NOTE: epochs in obtained data starts at 0)
                self.model_scores.append(model_score)

    def get_logs(self):
        """
        Returns dictionary containing experiment name, config dictionary, average training loss per epoch
        and model scores.
        """
        experiment_name = self.config['experiment']
        print(f'Model scores for "{experiment_name}": {self.model_scores}')
        return {'experiment': self.config['experiment'], 'config': self.config, 
                'average_training_loss_per_epoch': self.average_training_loss_per_epoch,
                'model_scores': self.model_scores}