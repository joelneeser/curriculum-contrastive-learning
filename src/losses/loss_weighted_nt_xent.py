""" Our main loss function.
i.e., NT-Xent (with cosine similarity) and sample reweighting
"""
import torch

from src.losses.cl_loss_pairwise_positives_IF import CLLossPairwisePositivesIF
from src.losses.similarities import cosine_similarity
from src.losses.weights import get_weight_object

EPS = 1e-10


class WeightedNTXent(CLLossPairwisePositivesIF):
    """ Our own loss, i.e., NT-Xent (with cosine similarity) and sample reweighting """

    def __init__(self, loss_config: dict) -> None:
        """
        Args:
            loss_config: See config.yaml for an example.
        """
        super().__init__()
        self.temp = loss_config["temperature"]
        self.weight = get_weight_object(loss_config["weight"],
                                        loss_config["weight_config"])
        self.weight_normalization = loss_config["weight_normalization"]

    def forward(self, pairwise_positives: torch.Tensor) -> torch.Tensor:
        """ Computes the total loss for our embeddings.
        Args:
            pairwise_positives: Embeddings, shape (2*K,D), where D is the feature
                dimension,and K>1 is the amount of positive pairs. The features are
                assumed to be listed in pairs, i.e. index 2*i and 2*i+1 form a
                positive pair for all i in [K].
        """

        #compute similarity matrix
        similarity_matrix = cosine_similarity(pairwise_positives)

        # Computation of weights for sample reweighting
        # Paper: The weights are meant to be constant w.r.t. the network. It
        #   shouldn't modify itself according to the "difficulty" of the sampled
        #   augmentations, only according to the resulting weighted loss.
        with torch.no_grad():
            weights = self.weight.get_weights(similarity_matrix)

        #compute (weighted) matrix of exponentials of similarities divided by temp
        exp_sim = torch.exp(similarity_matrix / self.temp)
        weighted_exp_sim = weights * exp_sim

        # exp_sim/weighted_exp_sim entries of positive pairs,
        # with indices (i, i + 1) for even i
        exp_sim_positives1 = torch.diagonal(exp_sim, offset=1)[::2]
        weighted_exp_sim_positives1 = torch.diagonal(weighted_exp_sim,
                                                     offset=1)[::2]
        # exp_sim/weighted_exp_sim entries of positive pairs,
        # with indices (i, i - 1) for odd i
        exp_sim_positives2 = torch.diagonal(exp_sim, offset=-1)[::2]
        weighted_exp_sim_positives2 = torch.diagonal(weighted_exp_sim,
                                                     offset=-1)[::2]

        #compute weighted denominators for all positive pairs
        intermediate_denominators = (torch.sum(weighted_exp_sim, dim=1) -
                                     torch.diagonal(weighted_exp_sim))

        if self.weight_normalization == "negatives_constant_total_weight":
            #computation of normalisation factors
            intermediate_sum_weights = (torch.sum(weights, dim=1) -
                                        torch.diagonal(weights))

            norm_factor_positives1 = (intermediate_sum_weights[::2] -
                                      torch.diagonal(weights, offset=1)[::2] +
                                      EPS) / (weights.shape[0] - 2 + EPS)

            norm_factor_positives2 = (intermediate_sum_weights[1::2] -
                                      torch.diagonal(weights, offset=-1)[::2] +
                                      EPS) / (weights.shape[0] - 2 + EPS)

            # Paper: "normalization" of second term in denominator of loss terms
            # -> keep all loss terms on a similar scale
            denominators_positives1 = ((
                (intermediate_denominators[::2] - weighted_exp_sim_positives1) /
                norm_factor_positives1) + exp_sim_positives1)

            denominators_positives2 = (
                ((intermediate_denominators[1::2] - weighted_exp_sim_positives2)
                 / norm_factor_positives2) + exp_sim_positives2)

        elif self.weight_normalization == "None":
            denominators_positives1 = (intermediate_denominators[::2] -
                                       weighted_exp_sim_positives1 +
                                       exp_sim_positives1)

            denominators_positives2 = (intermediate_denominators[1::2] -
                                       weighted_exp_sim_positives2 +
                                       exp_sim_positives2)
        else:
            raise NotImplementedError(
                f"Weight normalization {self.weight_normalization} hasn't " +
                "been implemented.")

        #Compute individual losses for positives1 and positives2
        losses1 = -torch.log(exp_sim_positives1 / denominators_positives1)
        losses2 = -torch.log(exp_sim_positives2 / denominators_positives2)

        #Compute final loss as mean of individual losses
        return (losses1.mean() + losses2.mean()) / 2

    def step(self) -> None:
        self.weight.step()
