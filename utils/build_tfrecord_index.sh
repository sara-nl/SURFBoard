#!/bin/bash -l 

#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --partition=gpu

SIMG=/home/xisurflp/cartesius_framework_java.sif
DATASET=/projects/2/managed_datasets/imagenet_tfrec_shuffled
INDEX=$PWD/tfrec_index

singularity exec $SIMG python build_tfrecord_index.py $DATASET/train $INDEX/train
singularity exec $SIMG python build_tfrecord_index.py $DATASET/validation $INDEX/validation
