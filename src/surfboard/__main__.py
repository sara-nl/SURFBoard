#!/usr/bin/env python3

import math
import argparse
import itertools
from enum import Enum
import subprocess
import sys
from abc import ABC, abstractmethod
import hydra
from copy import deepcopy
try:
    from omegaconf import Config
    to_container = Config.to_container
except:
    from omegaconf import OmegaConf
    to_container = OmegaConf.to_container

from surfboard.experiment import ProfileLevel
from surfboard.rn50_experiment import ResNet50Experiment
import surfboard
import os

config = os.path.dirname(surfboard.__file__)+'/../../configs/main.yaml'


class NeuralNetwork(Enum):
    RESNET50 = 'resnet50'
    

# Source: https://stackoverflow.com/a/40623158
def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def get_net_impl(net, **kwargs):
    if net == NeuralNetwork.RESNET50.value:
        return ResNet50Experiment
    else:
        raise Exception('Net {} not supported'.format(net))


def get_fixed_hparams(hparams):
    fixed = to_container(hparams, resolve=True)
    del fixed['net']
    del fixed['shared_sweep']
    del fixed['extra']
    del fixed['model_sweep']
    return fixed

@hydra.main(config_path=config)
def main(hparams):

    # Get the hparams that are sweeped over by all models/experiments
    sweep_hparams = to_container(hparams.shared_sweep, resolve=True)

    # Get the model/experiment implementation
    net_impl = get_net_impl(hparams.net)

    # Get possibly extra hparams sweeped over by a specific model
    model_specific_sweep_hparams = net_impl.model_specific_sweep_hparams(hparams.model_sweep)

    #Combine common sweeped over hparams and model specific ones
    sweep_hparams.update(model_specific_sweep_hparams)

    experiments = []
    for sweep_combination in dict_product(sweep_hparams):
        fixed_hparams = get_fixed_hparams(hparams)
        exp = net_impl(**fixed_hparams, **sweep_combination)
        if exp.is_legal():
            print(exp.cmd)
            print(exp.params)
            for i in range(hparams.extra.reps):
                subprocess.call("rm tauprofile.xml profile_*.sql", shell=True)
                retval = subprocess.call(exp.cmd, shell=True)
                if exp.profile_level.value == ProfileLevel.TAU_EXEC or \
                    exp.profile_level.value == ProfileLevel.TAU_PYTHON or \
                    exp.profile_level.value == ProfileLevel.TAU_PYTHON_CUDA: 
                    subprocess.call("mv tauprofile.xml run_" +
                                    str(i) +
                                    "_config_" +
                                    exp.params +
                                    "_ret_" +
                                    str(retval) + ".xml",
                                    shell=True)
                elif exp.profile_level.value == ProfileLevel.NVPROF or \
                    exp.profile_level.value == ProfileLevel.NSYS: 
                    subprocess.call("tar -cf run_" +
                                    str(i) +
                                    "_config_" +
                                    exp.params +
                                    "_ret_" +
                                    str(retval) + ".tar profile_*.sql",
                                    shell=True)
            experiments.append(exp)


if __name__ == '__main__':
    main()





