import os
from os import listdir
from os.path import isfile, join, exists, basename
from subprocess import call

import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import horovod.torch as hvd

DATA_BACKEND_CHOICES = ['pytorch']
IMAGE_FIELD_NAME = "image/encoded"
LABEL_FIELD_NAME = "image/class/label"

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.tfrecord as tfrec
    DATA_BACKEND_CHOICES.append('dali-gpu')
    DATA_BACKEND_CHOICES.append('dali-cpu')
    DATA_BACKEND_CHOICES.append('dali-cpu-to-gpu')
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


def tfrecord_input_dali_node(data_dir, index_dir, shuffle, rank, world_size):
    tfrecord_path_list = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    tfrecord_idx_path_list = [join(index_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    input = ops.TFRecordReader(path=tfrecord_path_list,
                               index_path=tfrecord_idx_path_list,
                               random_shuffle=shuffle,
                               shard_id=rank,
                               num_shards=world_size,
                               features={
                                   IMAGE_FIELD_NAME: tfrec.FixedLenFeature((), tfrec.string, ""),
                                   LABEL_FIELD_NAME: tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    return input


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, index_dir, crop, dali_cpu, tfrecord):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        local_rank = hvd.local_rank()
        world_size = hvd.size()
        self.tfrecord = tfrecord
        if tfrecord:
            self.input = tfrecord_input_dali_node(data_dir, index_dir, False, local_rank, world_size)
        else:
            self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = True)

        if dali_cpu:
            self.dali_device = "cpu"
            self.decode = ops.ImageDecoderRandomCrop(device=self.dali_device,
                                                    output_type=types.RGB,
                                                    random_aspect_ratio=[0.75, 4./3.],
                                                    random_area=[0.08, 1.0],
                                                    num_attempts=100)

        else:
            self.dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers
            # to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoderRandomCrop(device="mixed",
                                                      output_type=types.RGB,
                                                      device_memory_padding=211025920,
                                                      host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.75, 4./3.],
                                                      random_area=[0.08, 1.0],
                                                      num_attempts=100)

        self.res = ops.Resize(device=self.dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device = self.dali_device,
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        inputs = self.input(name = "Reader")
        if self.tfrecord:
            self.jpegs = inputs[IMAGE_FIELD_NAME]
            self.labels = inputs[LABEL_FIELD_NAME]
        else:
            self.jpegs, self.labels = inputs
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = images.gpu() if self.dali_device == 'gpu' else images
        output = self.cmnp(images, mirror = rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, index_dir, crop, size, dali_cpu, tfrecord):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        local_rank = hvd.local_rank()
        world_size = hvd.size()
        self.tfrecord = tfrecord
        if tfrecord:
            self.input = tfrecord_input_dali_node(data_dir, index_dir, False, local_rank, world_size)
        else:
            self.input = ops.FileReader(file_root=data_dir,
                                        shard_id=local_rank,
                                        num_shards=world_size,
                                        random_shuffle=False)
        if dali_cpu:
            self.dali_device = 'cpu'
            self.decode = ops.ImageDecoder(device=self.dali_device, output_type=types.RGB)
        else:
            self.dali_device = 'gpu'
            self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device = self.dali_device, resize_shorter = size)
        self.cmnp = ops.CropMirrorNormalize(device = self.dali_device,
                output_dtype = types.FLOAT,
                output_layout = types.NCHW,
                crop = (crop, crop),
                image_type = types.RGB,
                mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                std = [0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        inputs = self.input(name = "Reader")
        if self.tfrecord:
            self.jpegs = inputs[IMAGE_FIELD_NAME]
            self.labels = inputs[LABEL_FIELD_NAME]
        else:
            self.jpegs, self.labels = inputs
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = images.gpu() if self.dali_device == 'gpu' else images
        output = self.cmnp(images)
        return [output, self.labels]


class DALIWrapper(object):

    @staticmethod
    def gen_wrapper(dalipipeline, num_classes, one_hot, dali_cpu, to_gpu, tfrecord):
        for data in dalipipeline:
            input = data[0]["data"]
            target = data[0]["label"].squeeze()
            if tfrecord:
                target = target - 1  # This is because TF has ImageNet labels in [1, 1000] instead of [0, 999]
            if dali_cpu and to_gpu:
                input = input.cuda()
            if to_gpu:
                target = target.cuda().long()
            else:
                target = target.long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, num_classes, one_hot, dali_cpu, to_gpu, tfrecord):
        self.dalipipeline = dalipipeline
        self.num_classes =  num_classes
        self.one_hot = one_hot
        self.dali_cpu = dali_cpu
        self.to_gpu = to_gpu
        self.tfrecord = tfrecord

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline, self.num_classes, self.one_hot,
                                       self.dali_cpu, self.to_gpu, self.tfrecord)


def get_dali_train_loader(dali_cpu, to_gpu, tfrecord=True):
    def gdtl(data_path, index_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
        local_rank = hvd.local_rank()
        world_size = hvd.size()

        traindir = os.path.join(data_path, 'train')
        trainidxdir = os.path.join(index_path, 'train')

        pipe = HybridTrainPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = traindir, index_dir = trainidxdir,
                crop = 224, dali_cpu=dali_cpu, tfrecord=tfrecord)

        pipe.build()
        train_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(train_loader, num_classes, one_hot, dali_cpu, to_gpu, tfrecord), \
               int(pipe.epoch_size("Reader") / (world_size * batch_size))

    return gdtl


def get_dali_val_loader(dali_cpu, to_gpu, tfrecord=True):
    def gdvl(data_path, index_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
        local_rank = hvd.local_rank()
        world_size = hvd.size()

        valdir = os.path.join(data_path, 'validation')
        validxdir = os.path.join(index_path, 'validation')

        pipe = HybridValPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = valdir, index_dir = validxdir,
                crop = 224, size = 256, dali_cpu=dali_cpu, tfrecord=tfrecord)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(val_loader, num_classes, one_hot, dali_cpu, to_gpu, tfrecord), \
               int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdvl


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e

class PrefetchedWrapper(object):

    @staticmethod
    def prefetched_loader(loader, num_classes, fp16, one_hot):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if fp16:
            mean = mean.half()
            std = std.half()

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if fp16:
                    next_input = next_input.half()
                    if one_hot:
                        next_target = expand(num_classes, torch.half, next_target)
                else:
                    next_input = next_input.float()
                    if one_hot:
                        next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, num_classes, fp16, one_hot):
        self.dataloader = dataloader
        self.fp16 = fp16
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader, self.num_classes, self.fp16, self.one_hot)

def get_pytorch_train_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ]))

    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)

def get_pytorch_val_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                ]))

    val_sampler = torch.utils.data.RandomSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)
