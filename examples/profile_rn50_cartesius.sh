#!/bin/bash -l 

#SBATCH -t 2-12:00:00
#SBATCH --partition=gpu

SIMG=/projects/0/xisurf/xisurf.sif
SCAPS="--add-caps CAP_SYS_NICE"
RUN_CONF=single
PROFILE_LEVEL=tau_exec #tau_python for python profiling or tau_python_cuda for CUDA aware tau_python

DATASET=/projects/2/managed_datasets/imagenet_tfrec_shuffled
DATASET_INDEX=$PWD/tfrec_index
LIBSDIR=/home/xisurflp/xisurf/examples/tau/tmp_lib/

module purge
module load 2019
module load Python/3.6.6-foss-2018b
module unload OpenMPI/3.1.1-GCC-7.3.0-2.30
module load pre2019
module load mpi/openmpi/3.1.2-cuda10

export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/target_libs

#set TAU environment
TAU_HOME=/usr/local/x86_64
TAU_CONFIG=mpi-pthread-python-cupti-pdt-openmp-opari
export SINGULARITYENV_PREPEND_PATH=$TAU_HOME/bin
export SINGULARITYENV_TAU_MAKEFILE=$TAU_HOME/lib/Makefile.tau-$TAU_CONFIG
export SINGULARITYENV_PYTHONPATH=$TAU_HOME/lib/shared-$TAU_CONFIG
export SINGULARITYENV_TAU_PROFILE_FORMAT="merged"

#force ring allreduce
export SINGULARITYENV_OMPI_MCA_coll_tuned_use_dynamic_rules=1
export SINGULARITYENV_OMPI_MCA_coll_tuned_allreduce_algorithm=4
export SINGULARITYENV_OMPI_MCA_coll_base_verbose=1

#export TAU_THROTTLE=1
#export TAU_THROTTLE_NUMCALLS=3000
#export TAU_THROTTLE_PERCALL=5000
#export TAU_TRACK_MEMORY_FOOTPRINT=1

# Run a set of experiments defined by configs/rn50conf/${RUN_CONF}.yaml
python -m surfboard rn50conf=$RUN_CONF data_path=$DATASET index_path=$DATASET_INDEX max_num_nodes=8 profile_level=$PROFILE_LEVEL simg=$SIMG extra_sargs="-B /lib64:/host_libs -B /etc:/etc -B $LIBSDIR:/usr/lib64/ $SCAPS"
