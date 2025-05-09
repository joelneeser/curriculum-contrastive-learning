""" Weights and associated functions used by our weighted loss function. """

import random
import torch

####### Weights

## Helpers


class Weight_IF():
    """ Interface for the weight functions. """

    def get_weights(self, similarities: torch.Tensor) -> torch.Tensor:
        """ Return weights for given similarity matrix.

        Args:
            A (N, N)-shaped similarity matrix (symmetrical (N,N)- shaped tensor
            where entry (j,k) is the similarity between associated embeddings
            j and k
        Returns:
            Shape (N, N) tensor containing the corresponding weights.
        """
        raise NotImplementedError("A weight needs to implement get_weights.")

    def step(self) -> None:
        """ Update internal parameters after epoch end. """


def get_weight_object(weight_name: str, weight_config: dict) -> Weight_IF:
    """ Get a configured weight object by name. """
    try:
        return WEIGHTS[weight_name](weight_config)
    except KeyError as key:
        raise NotImplementedError(f"Weight function {key} is not implemented.")


## Weights


class GaussianWeight(Weight_IF):
    """ As described in our proposal in Eq. 2. """

    def __init__(self, weight_config: dict) -> None:
        """
        Args:
            weight_config: {"sigma": <sigma_value>,
                            "mean": <parameter_function_name>,
                            "mean_config": <config_dict_for_param_function>},
                            "b_l": <lower_similarity_bound>,
                            "b_u": <upper_similarity_bound>,
                where sigma > 0, -1 <= b_l < b_u <= 1
        """
        # Initialize parameters
        self.sigma = weight_config["sigma"]
        self.mean = get_varying_parameter(weight_config["mean"],
                                          weight_config["mean_config"])
        self.b_l = weight_config["b_l"]
        self.b_u = weight_config["b_u"]

    def get_weights(self, similarities: torch.Tensor) -> torch.Tensor:
        mean = self.mean.value
        weights = torch.exp(-(similarities - mean)**2 / self.sigma**2)
        weights[
            similarities < self.b_l] = 1e-3 * weights[similarities < self.b_l]
        weights[
            similarities > self.b_u] = 1e-3 * weights[similarities > self.b_u]
        return weights

    def step(self) -> None:
        self.mean.step()


WEIGHTS = {"GaussianWeight": GaussianWeight}

######## Time-varying parameters

## Helpers


class Parameter():
    """ Interface for the time-varying parameters. """

    def __init__(self) -> None:
        self.parameter = 1

    @property
    def value(self) -> float:
        """ The current value of the parameter. """
        return self.parameter

    def step(self) -> None:
        """ Update parameter after epoch end. """


def get_varying_parameter(name: str, config: dict) -> Parameter:
    """ Get a configured time-varying parameter by name. """
    try:
        return PARAMETERS[name](config)
    except KeyError as key:
        raise NotImplementedError(
            f"Parameter function {key} is not implemented.")


## Parameters

#for "positive" curriculum learning, choose ExponentialParam or LinearParam with begin < end.
#for "negative" curriculum learning, choose ExponentialParam or LinearParam with begin > end.
#for "random" curriculum learning, choose RandomParam with begin < end.

#TODO: better description of parameters?


class ExponentialParam(Parameter):
    """ An exponential evolution of a parameter from "begin" to "end". """

    def __init__(self, mean_config: dict) -> None:
        """
        Args:
            mean_config dict: See config.yaml for an example
        """
        super().__init__()

        self.factor = mean_config['factor']
        self._range = mean_config['begin'] - mean_config['end']
        self._exponential = 1
        self.parameter = mean_config['begin']
        self.end = mean_config['end']

    def step(self) -> None:
        self._exponential *= self.factor
        self.parameter = self.end + self._range * self._exponential


class LinearParam(Parameter):
    """ A linear evolution of a parameter from "begin" to "end".
        The parameter increases linearly from begin to end in "steps"
        many epochs, and then stays constant."
    """

    def __init__(self, mean_config: dict) -> None:
        """
        Args:
            mean_config dict: See config.yaml for an example
        """
        super().__init__()

        self.steps = mean_config['steps']
        self.interval_length = mean_config['end'] - mean_config['begin']
        self.parameter = mean_config['begin']
        self.end = mean_config['end']
        self.count = 0

    def step(self) -> None:
        if self.count < self.steps:
            self.parameter += self.interval_length / self.steps

        self.count += 1


class RandomParam(Parameter):
    """ A random evolution of a parameter in [begin, end].
        The parameter is drawn uniformly at random from the specified interval in every
        epoch.
    """

    def __init__(self, mean_config: dict) -> None:
        """
        Args:
            mean_config dict: See config.yaml for an example
        """
        super().__init__()

        self.interval_start = mean_config['begin']
        self.interval_end = mean_config['end']

        self.parameter = (self.interval_start - self.interval_end
                         ) * random.random() + self.interval_end

    def step(self) -> None:
        self.parameter = (self.interval_start - self.interval_end
                         ) * random.random() + self.interval_end


class RandomParamEveryIter(Parameter):
    """ A random evolution of a parameter in [begin, end].
        The parameter is drawn uniformly at random from the specified interval
        at every <iteration>.
    """

    def __init__(self, mean_config: dict) -> None:
        """
        Args:
            mean_config dict: See config.yaml for an example
        """
        super().__init__()

        self.interval_start = mean_config['begin']
        self.interval_end = mean_config['end']

        self.parameter = (self.interval_start - self.interval_end
                         ) * random.random() + self.interval_end

    def step(self) -> None:
        self.parameter = (self.interval_start - self.interval_end
                         ) * random.random() + self.interval_end

    @property
    def value(self) -> float:
        """ The current value of the parameter. """
        param = self.parameter
        self.step()
        return param


class RandomThenConst(Parameter):
    """ A random parameter at first, which then becomes constant. """

    def __init__(self, mean_config: dict) -> None:
        """
        Args:
            mean_config dict: {"begin":<min of random region>,
                               "end": <max of random region>,
                               "steps": <epoch at which value becomes constant>,
                               "const_value": <parameter value in constant region>}
        """
        super().__init__()
        self.random_param = RandomParam(mean_config)
        self.transition_epoch = mean_config["steps"]
        self.const_value = mean_config["const_value"]
        self.count = 0

        self.parameter = self.random_param.value

    def step(self) -> None:
        if self.count < self.transition_epoch:
            self.random_param.step()
            self.parameter = self.random_param.value
            self.count += 1
        else:
            self.parameter = self.const_value


PARAMETERS = {
    "ExponentialParam": ExponentialParam,
    "LinearParam": LinearParam,
    "RandomParam": RandomParam,
    "RandomThenConst": RandomThenConst,
    "RandomParamEveryIter":RandomParamEveryIter,
}
