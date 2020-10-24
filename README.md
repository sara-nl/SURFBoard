# SURFBoard
A framework for reproducible performance analysis of Deep Learning workflows

## Repo Structure

* Docker/Singularity image definitions in `container/`
* Python source code in `src/`
* NN Models for profiling in `models/`
    * Resnet50 in `models/pytorch/NvidiaResNet50` (Dali image pipeline, Horovod distribution, Apex for reduced precision)
* YAML configuration files in `configs/`
* An example SLURM job file in `examples/profile_rn50_cartesius.sh`, designed for the Cartesius system at SURFSara
* Utility scripts in `utils/`, for installing Hydra and building TFRecord index on Cartesius

## Installing and using the experiment orchestrator

```
pip install -e .
python -m surfboard
```

By default the orchestrator utilizes default settings in `configs/main.yaml`. Edit the YAML config for your own needs or provide overrides at the command line.

## Output profile data

Each experiment generates a single file of the format `run_[repindex]_[configstring]_ret_[retval].[xml|tar]`
* `repindex` is a number indicating the index of the specific repetition of the run configuration.  
* `confstring` reflects the experiment configuration and is structured as: `{run index}_config_{number of nodes}_{gpus per node}_{communication fabric}_{profile level}_{cpus per node}_{neural network}_{data loader type}_{batch size}_{gradient precision}_{compute precision}`
* `retval` is the return value of the profiling run
* extension is `xml` for TAU-based profiling runs and `tar` (containing multiple sql database files) for Nvidia profilers (NVProf/NSys) 

## Profiling ResNet50 on Cartesius

To profile the Nvidia ResNet50 design on Cartesius, launch a Slurm batch job as follows:
```
cd NvidiaResNet50
sbatch -N <nodes> profile_cartesius.sh
```
The batch script `profile_cartesius.sh` detects the number of available nodes and launches the experiment orchestrator appropriately.
The maximum number of GPUs per node is set to 2 and maximum number of CPU workers per GPU is set to 8 (reflecting node configuration on Cartesius).
Users can pass extra capabilities to Singularity containers by editing the SCAPS variable (currently set to enable CAP_SYS_NICE in the container).
Inspect the script for more details or to modify the location of the singularity image, dataset, and libraries on your system.
