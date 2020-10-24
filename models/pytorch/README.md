# Models

This directory contains various relevant model implementations.

## Pytorch

### NvidiaResNet50

This is an highly tuned implementation from Nvidia's DeepLearningExamples repository, modified to use Horovod instead of DDP for distributed training. 
- Horovod is supported for distributed training. Horovod+Apex integration performed according to [this issue](https://github.com/horovod/horovod/issues/1089) but not tested extensively.
- DALI on top of TFRecords is supported for data loading, possibly with GPU accelerated JPEG decoding and image preprocessing.

#### Setup

For ImageNet training with DALI, he implementation assumes that there is a */path/to/imagenet/folder* organized as follows:
- train (name of the folder is hardcoded)
    - *train-00000-of-****** (naming scheme of the files is irrelevant)
    - ...
- val (name of the folder is hardcoded)
    - *val-00000-of-****** (naming scheme of the files is irrelevant)
    - ...
   
where each tfrecord contains the following fields:
- *image/encoded* for the encoded jpeg
- *image/class/label* for the label

If required, change them at *NvidiaResNet50/image_classification/dataloaders.py*

#### Run

Run single GPU training with:

```
python main.py --arch resnet50 -c fanin --label-smoothing 0.1 /path/to/imagenet/tfrecord --data-backend dali-gpu
```

Run multi-GPU training with:

```
horovodrun -np 2 python main.py --arch resnet50 -c fanin --label-smoothing 0.1 /path/to/imagenet/tfrecord --data-backend dali-gpu
```

#### Known Issues

If training fails while attempting to allocate CUDA memory, reduce the batch size.
