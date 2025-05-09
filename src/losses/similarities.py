"""DEFINITIONS OF FUNCTIONS NEEDED IN TRAINING PROCESS. """

from typing import Callable
import torch

EPS = 1e-10


def cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """ Resturns the cosine similarity of all embeddings with each other.

    As used e.g. in the SimCLR paper.

    Args:
        embeddings: shape (N, D),  N is the number of embeddings, D the feature dimension.
    Returns:
        A (N, N)-shaped, symmetrical tensor, where entry (j,k) is the cosine
            similarity of embedding j and embedding k.
    """

    #Normalize the vectors
    norms = torch.linalg.vector_norm(embeddings.float(), dim=1, ord=2)
    embeddings_normalized = embeddings / (norms.unsqueeze(1) + EPS)

    #Return the similarity matrix
    return embeddings_normalized @ embeddings_normalized.T

### Helpers

SIMILARITIES = {
    "cosine_similarity": cosine_similarity
}


def get_similarity(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Get a similarity function by name. """
    try:
        return SIMILARITIES[name]
    except KeyError as key:
        raise NotImplementedError(f"Similarity {key} is not implemented.")
