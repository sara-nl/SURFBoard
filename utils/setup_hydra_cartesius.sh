#!/bin/bash -l 

#SBATCH -t 1:00:00
#SBATC -N 1
#SBATCH --partition=gpu

module purge
module load 2019
module load Python/3.6.6-foss-2018b

wget https://files.pythonhosted.org/packages/2c/ef/967df96458190a00337222a609dcfd01e886daa167fef6ef8ad4c0e93798/hydra-core-0.11.3.tar.gz
tar -xf hydra-core-0.11.3.tar.gz
cd hydra-core-0.11.3 && python setup.py install --user

