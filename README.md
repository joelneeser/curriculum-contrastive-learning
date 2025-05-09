# Curriculum Contrastive Learning of Image Representations via Sample Reweighting
*Joel Neeser, Jérôme Kuner, Jonas Mehr, Lukas Mouton — Group Project, ETH Zurich, 2022*

**Download the PDF** → [paper.pdf](./paper.pdf)

> **Abstract**
> How should we choose negative examples for contrastive learning? Early
> approaches simply sample uniformly at random from the training data, while more
> sophisticated methods attempt to pick suitable negative examples based on their
> hardness. We present a novel sample reweighting scheme that allows for varying
> the hardness of the considered negative examples as training progresses. It
> assesses the hardness of the possible negative examples and assigns weights
> based on how close they are to the currently desired hardness. Empirically, we
> find that varying the difficulty of negative examples during training does not
> improve performance in general, while focusing exclusively on semi-hard examples
> does. 

This repository contains the code to reproduce the results obtained in
"Curriculum Contrastive Learning of Image Representations via Sample
Reweighting".

## Reproducing the Results

1. Inside an activated virtual environment, confirm that all requirements in `requirements.txt` are satisfied, e.g., by running `pip install -r requirements.txt`. The code was tested with python 3 and the modules listed in `requirements.txt`.
2. Create a \*.yaml configuration file <config_name>.yaml and put it in `config/`. An example configuration file can be found at `config/config_example.yaml`.
3. To run the code with the given configuration file, pass the name of the config as an argument to `run.py` as follows: `python3 run.py <config_name>.yaml`. Make sure that this command is run from the base directory, where `run.py` is located.
4. The script will train and test the model and produce a \*.json file named `<experiment>.json` that will be saved to `experiment_logs/`, where \<experiment\> is the value that was given for experiment in the configuration file. The produced file contains several results from training and testing.

## Obtained Results

The logs of the experiments that we conducted and the corresponding configuration files can be found in `experiment_logs/` and `config/`, respectively. Running one such experiment on a GPU on the ETH EULER cluster with 4 CPU cores (4096 MB RAM per core, 16384 MB in total) should take approximately 7.5 hours.

## Configuration Parameters

This is a short explanation of the most important configuration parameters that were changed for the different experiments:

| parameter | function |
|---|---|
| pretraining: batch_size | sets the batch size used in training |
| pretraining: max_epochs | the number of epochs the model is trained for |
| loss: name | the name of the loss. Use WeightedNTXent for ours, NTXent for simCLR |
| temperature | the temperature parameter used in the loss |
| weight_normalization | whether to use weight normalization or not. Set to negatives_constant_total_weight to use it, None otherwise |
| sigma | the standard deviation of the Gaussian used to calculate the weights |
| mean | The schedule to use for moving the mean. Options: LinearParam, ExponentialParam, RandomParam, RandomThenConst, RandomParamEveryIter |
| begin | The weight mean used in the first epoch |
| end | The weight mean used after \<steps\> epochs |
| steps | The number of steps to reach the last value of the schedule |

## Licenses

All code in this repository is licensed under the **MIT License** (see [LICENSE](./LICENSE)). The paper ([paper.pdf](./paper.pdf)) is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license (see [LICENSE-paper](./LICENSE-paper)).
