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


class NetworkBackend(Enum):
    IB = 'ib'
    ETH = 'eth'


class ProfileLevel(Enum):
    DISABLED = None
    TAU_EXEC = 'tau_exec'
    TAU_PYTHON = 'tau_python'
    TAU_PYTHON_CUDA = 'tau_python_cuda'
    NSYS = 'nsys'
    NVPROF = 'nvprof'


class Experiment:

    def __init__(
            self,
            simg,
            num_nodes,
            max_num_nodes,
            gpus_per_node,
            omp_num_threads,
            network_backend,
            profile_level,
            extra_sargs,
            extra_mpiargs):

        # Int values
        self.simg = simg
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.omp_num_threads = omp_num_threads
        self.extra_sargs = extra_sargs
        self.extra_mpiargs = extra_mpiargs
        self.max_num_nodes = max_num_nodes

        # Strings that maps to enums
        self.network_backend = NetworkBackend(network_backend)
        self.profile_level = ProfileLevel(profile_level)


    @property
    @abstractmethod
    def cmd(self):
        pass

    @abstractmethod
    def is_legal(self):
        pass

    @staticmethod
    def model_specific_sweep_hparams(model_hparams):
        if model_hparams is None:
            return {}
        else:
            return model_hparams

    @property
    def params(self):
        parstr = f"{self.num_nodes}_" \
                 f"{self.gpus_per_node}_" \
                 f"{self.network_backend.value}_" \
                 f"{self.profile_level.value}_"
        return parstr

    @property
    def tot_gpus(self):
        return self.num_nodes * self.gpus_per_node

    @property
    def btl_type_cmd(self):
        if self.network_backend == NetworkBackend.ETH:
            return "self,tcp"
        elif self.network_backend == NetworkBackend.IB:
            return "^tcp..."

    @property
    def profile_level_cmd(self):
        if self.profile_level == ProfileLevel.DISABLED:
            return "python"
        elif self.profile_level == ProfileLevel.TAU_PYTHON:
            return "tau_python"
        elif self.profile_level == ProfileLevel.TAU_PYTHON_CUDA:
            return "tau_python -T cupti -cupti"
        elif self.profile_level == ProfileLevel.TAU_EXEC:
            return "tau_exec -T pthread python"
        elif self.profile_level == ProfileLevel.NSYS:
            return "nsys profile -t nvtx,osrt,cuda --sample=cpu --output profile_%q{OMPI_COMM_WORLD_RANK} python"
        elif self.profile_level == ProfileLevel.NVPROF:
            return "nvprof -f -o profile_%q{OMPI_COMM_WORLD_RANK}.sql python"

    @property
    @abstractmethod
    def cmd_app(self):
        pass

    @property
    def cmd_prefix(self):
        cmd = f"OMP_NUM_THREADS={self.omp_num_threads} mpirun " \
              f"--map-by ppr:{self.gpus_per_node}:node " \
              f"-np {self.tot_gpus} " \
              f"-x OMP_NUM_THREADS " \
              f"--mca btl {self.btl_type_cmd} {self.extra_mpiargs} " \
              f"singularity exec --nv {self.extra_sargs} -B {self.data_path}:/dataset -B {self.index_path}:/index {self.simg} " \
              f"{self.profile_level_cmd} "
        return cmd 

    @property
    def cmd(self):
        cmd = f"{self.cmd_prefix }" \
              f"{self.cmd_app}"
        return cmd
