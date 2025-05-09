""" Our implementation of the loss function NT-Xent (with cosine similarity) used in the SimCLR paper.
"""

import torch

from src.losses.cl_loss_pairwise_positives_IF import CLLossPairwisePositivesIF
from src.losses.similarities import cosine_similarity

class NTXent(CLLossPairwisePositivesIF):
    """ NT-Xent with cosine similarity (the loss function used in SimCLR)"""

    def __init__(self, loss_config: dict) -> None:
        """
        Args:
            loss_config: {"temperature": <temperature>}
        """
        super().__init__()
        self.temp = loss_config["temperature"]

    
    def forward(self, pairwise_positives: torch.Tensor) -> torch.Tensor:
        """ Computes the total loss for our embeddings.
        Args:
            pairwise_positives: Embeddings, shape (2*K,D), where D is the dimension
                of the latent space, and K is the amount of positive pairs. The features 
                are assumed to be listed in pairs, i.e. index 2*i and 2*i+1 form a
                positive pair for all i in {0,...,K-1}.
        """
        
        #compute similarity matrix
        similarity_matrix = cosine_similarity(pairwise_positives)

        #compute matrix of exponentials of similarities divided by temp
        exp_sim = torch.exp(similarity_matrix / self.temp)

        #compute denominators (sums of all exp(sim), for all positive pairs, both directions)
        denominators = (torch.sum(exp_sim, dim=1) - torch.diagonal(exp_sim))

        #exp(similarities) of the positive pairs
        # (i, i + 1 | for even i)
        exp_sim_positives1 = torch.diagonal(exp_sim, offset=1)[::2]
        # (i, i - 1 | for odd i)
        exp_sim_positives2 = torch.diagonal(exp_sim, offset=-1)[::2]

        #Compute individual l(i,j) loss terms
        losses1 = -torch.log(exp_sim_positives1 / denominators[::2])
        losses2 = -torch.log(exp_sim_positives2 / denominators[1::2])

        #compute final loss as mean of individual losses
        return (losses1.mean() + losses2.mean()) / 2
    

