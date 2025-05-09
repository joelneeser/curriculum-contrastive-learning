""" This module implements the loss interface for contrastive learning.

It is more similar to the SimCLR setting, i.e., we have pairwise positive features,
and assume nothing else about all other features.
"""
import torch
from torch import nn


class CLLossPairwisePositivesIF(nn.Module):
    """Specific Loss Interface for Contrastive Learning."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pairwise_positives: torch.Tensor) -> torch.Tensor:
        """ Returns the <total> loss for the given positive pairs.

        Args:
            pairwise_positives: shape (2*K,D), where D is the feature dimension,
                and K is the amount of positive pairs. The features are assumed
                to be listed in pairs, i.e. index 2*i and 2*i+1 form a positive
                pair for all i in [K].
        """
        raise NotImplementedError("A loss must implement a forward method")

    def step(self) -> None:
        """ Updates the state of the loss after an epoch has elapsed.

        If updates after every batch are desired, these can be placed in the
        forward function.
        """
