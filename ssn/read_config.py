import models
from trainer import losses as losses
import torch.nn
from torch.utils.data.dataloader import DataLoader
from nifti.datasets import PatchWiseNiftiDataset, FullImageToOverlappingPatchesNiftiDataset, worker_init_fn
from nifti.savers import NiftiPatchSaver
import nifti.transformation
import nifti.augmention
from nifti import patch_samplers
from trainer import metrics as trainer_metrics
from trainer.metrics import Loss
from trainer.metrics import SegmentationMetrics, SegmentationImageThumbs
from trainer.hooks import TrainingEvaluator, ValidationEvaluator, ModelSaverHook, NaNLoss


def get_augmentation(augmentation_dict):
    return [getattr(nifti.augmention, name)(**kwargs) for name, kwargs in augmentation_dict.items()]


def get_transformation(transformation_dict):
    return [getattr(nifti.transformation, name)(**kwargs) for name, kwargs in transformation_dict.items()]


def get_patch_wise_dataset(config, model, csv_path, train=True):
    key = 'training' if train else 'valid'
    input_patch_size = tuple(config[key]['input_patch_size'])
    output_patch_size = model.get_output_size(input_patch_size)

    transformation = get_transformation(config['data']['transformation'])
    augmentation = get_augmentation(config['training']['augmentation']) if train else []
    patch_augmentation = get_augmentation(config['training']['patch_augmentation']) if train else []

    # same sampling used for train and validation to ensure comparable curves
    sampler_type = list(config['training']['sampler'].keys())[0]
    config['training']['sampler'][sampler_type].update({'augmentation': patch_augmentation})
    sampler_class = getattr(patch_samplers, sampler_type)
    sampler = sampler_class(input_patch_size, output_patch_size, **config['training']['sampler'][sampler_type])

    sampling_mask = config['data']['sampling_mask'] if 'sampling_mask' in config['data'] else None
    sample_weight = config['data']['sample_weight'] if 'sample_weight' in config['data'] else None

    dataset = PatchWiseNiftiDataset(patch_sampler=sampler,
                                    images_per_epoch=config[key]['images_per_epoch'],
                                    patches_per_image=config[key]['patches_per_image'],
                                    data_csv_path=csv_path,
                                    channels=config['data']['channels'],
                                    target=config['data']['target'],
                                    sampling_mask=sampling_mask,
                                    sample_weight=sample_weight,
                                    transformation=transformation,
                                    augmentation=augmentation,
                                    max_cases_in_memory=config[key]['max_cases_in_memory'],
                                    sequential=False if train else True)
    return dataset


def get_train_loader(config, model, train_csv_path, use_cuda):
    train_set = get_patch_wise_dataset(config, model, train_csv_path)
    train_loader = DataLoader(train_set,
                              batch_size=config['training']['batch_size'],
                              num_workers=config['training']['num_workers'],
                              worker_init_fn=worker_init_fn,
                              pin_memory=True if use_cuda else False)
    return train_loader


def get_valid_loader(config, model, test_csv_path, use_cuda):
    valid_set = get_patch_wise_dataset(config, model, test_csv_path, train=False)
    valid_loader = DataLoader(valid_set,
                              batch_size=config['valid']['batch_size'],
                              num_workers=config['valid']['num_workers'],
                              worker_init_fn=worker_init_fn,
                              pin_memory=True if use_cuda else False)

    return valid_loader


def get_test_loader(config, model, test_csv_path, use_cuda):
    if config['test'] is None:
        return None
    transformation = get_transformation(config['data']['transformation'])
    sampling_mask = config['data']['sampling_mask'] if 'sampling_mask' in config['data'] else None
    input_patch_size = tuple(config['test']['input_patch_size'])
    output_patch_size = model.get_output_size(input_patch_size)
    use_bbox = config['test']['use_bbox'] if 'use_bbox' in config['test'] else False
    test_set = FullImageToOverlappingPatchesNiftiDataset(image_patch_shape=input_patch_size,
                                                         target_patch_shape=output_patch_size,
                                                         data_csv_path=test_csv_path,
                                                         channels=config['data']['channels'],
                                                         target=config['data']['target'],
                                                         sampling_mask=sampling_mask,
                                                         transformation=transformation,
                                                         use_bbox=use_bbox)

    test_loader = DataLoader(test_set,
                             batch_size=config['test']['batch_size'],
                             shuffle=False,
                             num_workers=config['test']['num_workers'],
                             worker_init_fn=worker_init_fn,
                             pin_memory=True if use_cuda else False)
    return test_loader


def get_metrics(device, config):
    metrics = {'loss': Loss(device),
               'metrics': SegmentationMetrics(device, config['data']['class_names']),
               'thumbs': SegmentationImageThumbs(config['training']['batch_size'])}
    if 'extra_metrics' in config['training']:
        for name, extra_metric in config['training']['extra_metrics'].items():
            key = list(extra_metric.keys())[0]
            metrics.update({name: getattr(trainer_metrics, key)(**extra_metric[key])})
    return metrics


def get_training_hooks(job_dir, config, device, valid_loader, test_loader):
    hooks = [TrainingEvaluator(job_dir + '/train', get_metrics(device, config)),
             NaNLoss(),
             ModelSaverHook(config['valid']['eval_every'], config['valid']['keep_model_every'])]

    if valid_loader is not None:
        hooks.append(ValidationEvaluator(job_dir + '/val', get_metrics(device, config), valid_loader,
                                         config['valid']['eval_every']))

    if test_loader is not None:
        extra_output_names = config['test']['extra_output_names'] if 'extra_output_names' in config['test'] else None
        saver = NiftiPatchSaver(job_dir + '/test', test_loader, extra_output_names=extra_output_names)
        hooks.append(ValidationEvaluator(job_dir + '/test',
                                         get_metrics(device, config),
                                         test_loader,
                                         config['test']['eval_every'],
                                         saver))
    return hooks


def get_model(config):
    model_type = list(config['model'].keys())[0]
    model_class = getattr(models, model_type)
    model = model_class(**config['model'][model_type])
    return model


def get_loss(config):
    loss_type = list(config['loss'].keys())[0]
    loss_class = getattr(losses, loss_type)
    loss = loss_class(**config['loss'][loss_type])
    return loss


def get_optimizer(config, model):
    optimizer_type = list(config['optimizer'].keys())[0]
    optimizer_class = getattr(torch.optim, optimizer_type)
    optimizer = optimizer_class(model.parameters(), **config['optimizer'][optimizer_type])
    scheduler_type = list(config['scheduler'].keys())[0]
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
    scheduler = scheduler_class(optimizer, **config['scheduler'][scheduler_type])
    return scheduler
