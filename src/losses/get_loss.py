from src.losses.loss_weighted_nt_xent import WeightedNTXent
from src.losses.loss_nt_xent import NTXent
LOSSES = {"WeightedNTXent": WeightedNTXent, "NTXent": NTXent}


def get_loss(loss_dict: dict):
    """ Returns the loss by its loss dict, e.g., 
        {'name': 'NTXent', 'loss_config': {'temperature': 1}}
    or see config.yaml.
    """

    try:
        return LOSSES[loss_dict["name"]](loss_dict["loss_config"])
    except KeyError as key:
        raise NotImplementedError(f"Loss {key} is not implemented.")
