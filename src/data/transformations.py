""" This module implements the transforms used in our model."""

from typing import Dict, Tuple, Union
from torchvision.transforms import RandomApply, Compose
from torchvision.transforms import (RandomResizedCrop, RandomHorizontalFlip,
                                    ColorJitter, RandomGrayscale, GaussianBlur,
                                    ToTensor)

#All implemented transforms
TRANSFORMS = {
    "RandomResizedCrop": RandomResizedCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "ColorJitter": ColorJitter,
    "RandomGrayscale": RandomGrayscale,
    "GaussianBlur": GaussianBlur
}


def get_transform(transform_config: Dict[str, dict]):
    """ Given a dict of transform dicts, returns the composed transform.
    
    Args:
        Dictionary of dictionaries containing torchvision transform configurations as described
        in parse_config_dict, e.g.
            {'RandomResizedCrop': {'size': '32,32', 'scale': '0.08,1.0', 'ratio': '0.75,1.3333333333'},
            'RandomHorizontalFlip': {'p': 0.5}}

    Returns:
        Composition of all transforms and a final toTensor() transform
    """
    transform_list = []
    for transform_name, transform_params in transform_config.items():
        try:
            transform_constructor = TRANSFORMS[transform_name]
        except KeyError as e:
            raise NotImplementedError(
                f"The transform/data augmentation {e} is not implemented.")
        cleaned_transform_params, prob = parse_config_dict(transform_params)
        if prob is None:
            transform_list.append(
                transform_constructor(**cleaned_transform_params))
        else:
            transform_list.append(
                RandomApply([transform_constructor(**cleaned_transform_params)],
                            p=prob))
    transform_list.append(ToTensor())
    return Compose(transform_list)


def parse_config_dict(config: dict) -> Tuple[dict, Union[float, None]]:
    """ Prepares transform_config for use in get_transform.

    Args:
        Dictionary containing configuration for a torchvision transform, e.g., 
            {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1, 'prob': 0.8}
        for the transform ColorJitter, or
            {'size': '32,32', 'scale': '0.08,1.0', 'ratio': '0.75,1.3333333333'}
        for the transform RandomResizedCrop

    Returns:
        ("Cleaned" dictionary, probability parameter) 
    
    The cleaned dictionary has all numerical values as floats, and sequences of numerical values as 
    tuples of floats. The probability parameter denotes the probability that the transform is applied.
    The probability parameter is None if the transform should always be applied. For example, for the 
    transform ColorJitter we get
            ({'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}, 0.8)
    """
    # Parse the dict representations
    cleaned_dict = {}
    prob = None
    for param_name, param_value in config.items():
        if param_name == "prob":
            prob = param_value
        elif isinstance(param_value, str):
            string_list = param_value.split(",")
            cleaned_param_value = tuple(stringToNumerical(str_int) for str_int in string_list)
            cleaned_dict.update({param_name: cleaned_param_value})
        else:
            cleaned_dict.update({param_name: param_value})

    return cleaned_dict, prob


def stringToNumerical(string: str):
    """" Needed for conversion of numerical value encoded in string to int/float in parse_config_dict.
    Args: string of numerical value, e.g., '2.1' or '2'

    Returns: Corresponding value as an integer if it is an integer, and as a float otherwise.
    """
    try:
        return int(string)
    except ValueError:
        return float(string)
