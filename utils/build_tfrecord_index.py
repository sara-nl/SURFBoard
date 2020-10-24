import os
from os import listdir
from os.path import isfile, join, exists, basename
from subprocess import call
import sys

if len(sys.argv) < 2:
    print("Please provide dataset folder path and optionally index folder path")
    exit(-1)
else:
    data_dir = sys.argv[1]

if len(sys.argv) < 3:
    idx_files_dir = join(os.getcwd(), "idx_files")
else:
    idx_files_dir = sys.argv[2]

tfrecord_path_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]

if not exists(idx_files_dir):
    os.mkdir(idx_files_dir)
tfrecord_idx_path_list = []
for tfrecord_path in tfrecord_path_list:
    tfrecord_name = basename(tfrecord_path)
    tfrecord_idx_path = join(idx_files_dir, tfrecord_name)
    tfrecord_idx_path_list.append(tfrecord_idx_path)
    print("Processing tfrecord"+tfrecord_path)
    if not isfile(tfrecord_idx_path):
        print("Building index file "+tfrecord_idx_path)
        call(["tfrecord2idx", tfrecord_path, tfrecord_idx_path])

