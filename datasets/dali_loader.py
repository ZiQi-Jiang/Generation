import math
import numpy as np
from nvidia.dali import fn, types, pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.auto_aug.rand_augment import rand_augment
from nvidia.dali.fn.random import uniform, coin_flip

def dino_view(images, size, scale, blur_p, solorize_p):
    x = fn.random_resized_crop(images, device='gpu', size=[size, size], random_area=scale, random_aspect_ratio=[0.75, 4/3], num_attempts=10, interp_type=types.INTERP_CUBIC, antialias=True)
    x = fn.flip(x, device='gpu', horizontal=coin_flip(probability=0.5))
    if coin_flip(probability=0.8):
        x = fn.color_twist(x, device='gpu', brightness=uniform(range=[0.6, 1.4]), contrast=uniform(range=[0.6,1.4]), saturation=uniform(range=[0.8, 1.2]), hue=uniform(range=[-36.0, 36.0]))

    if coin_flip(probability=0.2):
        gray = fn.color_space_conversion(x, device='gpu', image_type=types.RGB, output_type=types.GRAY)
        x = fn.cat(gray, gray, gray, axis=2, device='gpu')

    if coin_flip(probability=blur_p):
        x = fn.gaussian_blur(x, device='gpu', sigma=uniform(range=[0.1, 2.0]))

    if coin_flip(probability=solorize_p):
        x = fn.cast( (x < 128) * x + (x >= 128) * (255-x), dtype=types.UINT8, device='gpu')

    return fn.crop_mirror_normalize(x, device='gpu', dtype=types.FLOAT, output_layout='CHW', 
                                    mean=[255*v for v in [0.485, 0.456, 0.406]], std = [255*v for v in [0.229, 0.224, 0.225]])


@pipeline_def(enable_conditionals=True)
def dino_pipeline(files, labels, rank, world_size, cfg):
    encoded, target = fn.readers.file(files=files, labels=labels, shard_id=rank, num_shards=world_size, shuffle_after_epoch=True, shuffle_after_epoch_seed=cfg.seed, seed=cfg.seed, pad_last_batch=True, name='Reader')
    images = fn.decoders.image(encoded, device='mixed', output_type=types.RGB)
    views = [dino_view(images, 224, cfg.global_crops_scale, 1.0, 0.0), 
             dino_view(images, 224, cfg.global_crops_scale, 0.1, 0.2)]

    views += [dino_view(images, 96, cfg.local_crops_scale, 0.5, 0.0) for _ in range(cfg.local_views)]
    return (*views, target.gpu())

class DinoDaliLoader:
    def __init__(self, dataset, cfg, accelerator):
        order = np.random.default_rng(cfg.seed).permutation(len(dataset))
        files = [dataset.samples[i][0] for i in order]
        labels = [dataset.samples[i][1] for i in order]

        self.view_names = [f'view{i}' for i in range(2+cfg.local_views)]
        self.num_batches = len(dataset) //(cfg.batch_size * accelerator.num_processes)
        pipeline = dino_pipeline(files=files, labels=labels, rank=accelerator.process_index, world_size=accelerator.num_processes, cfg=cfg, batch_size=cfg.batch_size, device_id=accelerator.device.index, num_threads=4, prefetch_queue_depth=2, seed=cfg.seed + accelerator.process_index)
        self.iterator = DALIGenericIterator([pipeline], output_map=self.view_names+['label'], reader_name='Reader', auto_reset=True, last_batch_policy=LastBatchPolicy.FILL, prepare_first_batch=False)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for step, result in enumerate(self.iterator):
            if step < self.num_batches:
                batch = result[0]
                views = [batch[name] for name in self.view_names]
                labels = batch['label'].reshape(-1).long()
                yield views, labels


@pipeline_def
def jit_pipeline(files, labels, rank, world_size, cfg):
    encoded, target = fn.readers.file(files=files, labels=labels, shard_id=rank, num_shards=world_size, shuffle_after_epoch=True, shuffle_after_epoch_seed=cfg.seed, seed=cfg.seed, pad_last_batch=True, name='Reader')

    images = fn.decoders.image(encoded, device='mixed', output_type=types.RGB)
    images = fn.resize(images, device='gpu', resize_shorter=cfg.image_size, interp_type=types.INTERP_CUBIC, antialias=True)
    images = fn.crop_mirror_normalize(images, device='gpu', crop=(cfg.image_size, cfg.image_size), crop_pos_x=0.5, crop_pos_y=0.5, 
                                      mirror=coin_flip(probability=0.5), dtype=types.UINT8, output_layout='CHW', rounding='truncate')
    return images, target.gpu()

class JitDaliLoader:
    def __init__(self, dataset, cfg, accelerator):
        order = np.random.default_rng(cfg.seed).permutation(len(dataset))
        files = [dataset.samples[i][0] for i in order]
        labels = [dataset.samples[i][1] for i in order]   
        self.num_batches = len(dataset) //(cfg.batch_size * accelerator.num_processes)

        pipeline = jit_pipeline(files=files, labels=labels, rank=accelerator.process_index, world_size=accelerator.num_processes, cfg=cfg, batch_size=cfg.batch_size, device_id=accelerator.device.index, num_threads=4, prefetch_queue_depth=2, seed=cfg.seed + accelerator.process_index)
        self.iterator = DALIGenericIterator([pipeline], output_map=['image', 'label'], reader_name='Reader', auto_reset=True, last_batch_policy=LastBatchPolicy.FILL, prepare_first_batch=False)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for step, result in enumerate(self.iterator):
            if step < self.num_batches:
                batch = result[0]
                yield batch['image'], batch['label'].reshape(-1).long()