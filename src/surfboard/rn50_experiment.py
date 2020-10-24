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

from surfboard.experiment import Experiment
import surfboard
import os

rn50main = os.path.dirname(surfboard.__file__)+'/../../models/pytorch/NvidiaResNet50/main.py'

class DataFormat(Enum):
    FOLDER = 'folder'
    TFRECORD = 'tfrecord'


class DataLoader(Enum):
    PYTORCH = 'pytorch'
    DALI_CPU_TO_GPU = 'dali-cpu-to-gpu'
    DALI_GPU = 'dali-gpu'


class DistributedBackend(Enum):
    HOROVOD = 'horovod'


class ComputePrecision(Enum):
    MIXED = 'mixed'
    FP16 = 'fp16'
    FP32 = 'fp32'


class GradPrecision(Enum):
    FP16 = 'fp16'
    FP32 = 'fp32'


class ResNet50Experiment(Experiment):

    def __init__(
            self,
            data_format,
            data_loader,
            data_path,
            index_path,
            simg,
            distributed_backend,
            compute_precision,
            grad_precision,
            num_nodes,
            max_num_nodes,
            max_workers,
            gpus_per_node,
            workers,
            omp_num_threads,
            batch_size_per_gpu,
            network_backend,
            profile_level,
            extra_sargs,
            extra_mpiargs,
            batches_per_rep):

        super().__init__(
            simg,
            num_nodes,
            max_num_nodes,
            gpus_per_node,
            omp_num_threads,
            network_backend,
            profile_level,
            extra_sargs,
            extra_mpiargs)

        # Int values
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batches_per_rep = batches_per_rep
        self.max_workers = max_workers
        self.workers = workers

        # Strings that maps to enums
        self.data_format = DataFormat(data_format)
        self.data_loader = DataLoader(data_loader)
        self.distributed_backend = DistributedBackend(distributed_backend)
        self.compute_precision = ComputePrecision(compute_precision)
        self.grad_precision = GradPrecision(grad_precision)

    def is_legal(self):
        if self.data_format == DataFormat.TFRECORD and self.data_loader == DataLoader.PYTORCH:
            print("Illegal data format (TFrecord with Pytorch)")
            return False
        if self.max_num_nodes is not None and self.num_nodes > self.max_num_nodes:
            print("Illegal num_nodes")
            return False
        if self.max_workers is not None and self.workers > self.max_workers:
            print("Illegal max_workers")
            return False
        return True

    @property
    def grad_precision_cmd(self):
        if self.grad_precision == GradPrecision.FP16:
            return "--fp16-allreduce"
        elif self.grad_precision == GradPrecision.FP32:
            return ""

    @property
    def compute_precision_cmd(self):
        if self.compute_precision == ComputePrecision.FP32:
            return ""
        elif self.compute_precision == ComputePrecision.MIXED:
            return "--amp"
        elif self.compute_precision == ComputePrecision.FP16:
            return "--fp16"
        else:
            return self.compute_precision.value

    @property
    def params(self):
        parstr = f"{self.workers}_" \
                 f"{self.data_loader.value}_" \
                 f"{self.batch_size_per_gpu}_" \
                 f"{self.grad_precision.value}_" \
                 f"{self.compute_precision.value}"
        return super().params + parstr

    @property
    def cmd_app(self):
        cmd = f"{rn50main} " \
              f"--workers {self.workers} " \
              f"--arch resnet50 " \
              f"-c fanin " \
              f"/dataset " \
              f"--dataidx /index " \
              f"--data-backend {self.data_loader.value} " \
              f"-b {self.batch_size_per_gpu} " \
              f"{self.grad_precision_cmd} " \
              f"{self.compute_precision_cmd} " \
              f"--prof {self.batches_per_rep} --no-checkpoints --epochs 1 --training-only"
        return cmd

