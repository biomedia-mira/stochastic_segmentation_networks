from evaluation.running_metrics.running_metric import RunningMetric
import torch
import inspect
import numpy as np
import math
import SimpleITK as sitk


def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[np.isnan(iou_)] = 1.
    return iou_


# exclude background
def distance(x, y):
    try:
        per_class_iou = iou(x[:, None], y[None, :], axis=-2)
    except MemoryError:
        per_class_iou = []
        for x_ in x:
            per_class_iou.append(iou(np.expand_dims(x_, axis=0), y[None, :], axis=-2))
        per_class_iou = np.concatenate(per_class_iou)
    return 1 - per_class_iou[..., 1:].mean(-1)


# # exclude background - not vectorized version
# def distance_(x, y):
#     per_class_iou = []
#     for x_ in x:
#         tmp = []
#         for y_ in y:
#             tmp.append(iou(x_, y_, axis=-2))
#         per_class_iou.append(np.stack(tmp))
#     per_class_iou = np.stack(per_class_iou)
#     return 1 - per_class_iou[..., 1:].mean(-1)
#
#
# def vectorized_generalised_energy_distance(sample_arr, gt_arr):
#     sample_arr = sample_arr.reshape((sample_arr.shape[0], -1, sample_arr.shape[-1]))
#     gt_arr = gt_arr.reshape((gt_arr.shape[0], -1, gt_arr.shape[-1]))
#     diversity = np.mean(distance(sample_arr, sample_arr))
#     ged = 2 * np.mean(distance(sample_arr, gt_arr)) - diversity - np.mean(distance(gt_arr, gt_arr))
#     return ged, diversity


def calc_sample_diversity(samples, num_samples_to_use):
    samples = samples[:num_samples_to_use]
    return np.mean(distance(samples, samples))


def calc_generalised_energy_distance(samples_dist_0, samples_dist_1, num_classes, num_samples_to_use):
    samples_dist_0 = samples_dist_0[:min(num_samples_to_use, len(samples_dist_0))]
    samples_dist_1 = samples_dist_1[:min(num_samples_to_use, len(samples_dist_1))]
    samples_dist_0 = samples_dist_0.reshape((len(samples_dist_0), -1))
    samples_dist_1 = samples_dist_1.reshape((len(samples_dist_1), -1))
    eye = np.eye(num_classes)
    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)

    cross = np.mean(distance(samples_dist_0, samples_dist_1))
    diversity_0 = np.mean(distance(samples_dist_0, samples_dist_0))
    diversity_1 = np.mean(distance(samples_dist_1, samples_dist_1))
    return 2 * cross - diversity_0 - diversity_1, diversity_0, diversity_1


def calc_marginal_entropy(prob_map):
    return -torch.sum(prob_map * torch.log(prob_map) / math.log(prob_map.shape[0]), axis=0)


# labels should be one hot
def calc_loglikelihood(labels, prob_maps):
    m = prob_maps.shape[0]
    labels = labels.permute((3, 0, 1, 2)).type(prob_maps.dtype)
    a = torch.einsum('cijk,ncijk->n', labels, prob_maps.log() + 1e-10)
    return torch.logsumexp(a) - torch.log(m)
    # return torch.logsumexp(torch.sum(labels * prob_maps.log(), axis=(1, 2, 3, 4)), axis=0) - torch.log(m)


class RunningDistributionStatistics(RunningMetric):
    def __init__(self, classes, require_prob_maps, extra_maps, num_samples, sampler_class, sampler_kwargs,
                 block_size=2, num_samples_cap=20):
        super().__init__(classes, require_prob_maps, extra_maps)
        self.num_classes = len(classes)
        self.num_samples = num_samples
        self.sampler_class = sampler_class
        self.sampler_kwargs = sampler_kwargs
        self.block_size = block_size
        self.num_samples_cap = num_samples_cap  # number of samples to keep / calculate diversity (memory constraints)
        self.eye = torch.eye(self.num_classes, self.num_classes, device=sampler_kwargs['device'])
        self.cms = []
        self.ged = []
        self.diversity = []
        self.loglikelihood = []
        self.average_pixel_wise_entropy = []

    def compute_confusion_matrix(self, segmentation, samples):
        return torch.einsum('nd,bne->bde', self.eye[segmentation.flatten()], self.eye[samples.view(samples.shape[0], -1)])

    def _evaluate(self, segmentation, prediction, prob_maps, mask, extra_maps):
        torch.cuda.empty_cache()
        torch.no_grad()
        extra_maps = extra_maps.copy()
        extra_maps.update({'prob_maps': prob_maps, 'mask': mask})
        params = inspect.signature(self.sampler_class.__dict__['__init__']).parameters.keys()
        kwargs = {key: value for key, value in extra_maps.items() if key in params}
        kwargs.update(self.sampler_kwargs)
        sampler = self.sampler_class(**kwargs)

        segmentation = torch.tensor(segmentation, dtype=torch.int64, device=self.sampler_kwargs['device'], requires_grad=False)
        one_hot_segmentation = self.eye[segmentation].permute((3, 0, 1, 2)).type(torch.float64)

        cm = []
        avg_sample_prob_map = None
        samples = []
        sitk_samples = []
        loglikelihood = []
        sample_prob_maps = None

        # must be done in blocks due to memory
        for _ in range(self.num_samples // self.block_size):
            sitk_samples_, samples_, prob_map_ = sampler(num_samples=self.block_size)
            cm.append(self.compute_confusion_matrix(segmentation, samples_))
            loglikelihood.append(torch.sum(prob_map_[:, one_hot_segmentation.type(torch.bool)].log(), axis=-1))
            prob_map_ = torch.sum(prob_map_, axis=0)
            avg_sample_prob_map = avg_sample_prob_map + prob_map_ if avg_sample_prob_map is not None else prob_map_
            samples += list(samples_.cpu().numpy().astype(np.uint8))
            sitk_samples += sitk_samples_
            del samples_
        del sampler
        loglikelihood = torch.cat(loglikelihood)
        loglikelihood = torch.logsumexp(loglikelihood, axis=0) - math.log(len(loglikelihood))
        loglikelihood = loglikelihood.cpu().numpy().astype(np.float32)
        avg_sample_prob_map = avg_sample_prob_map / self.num_samples + 1e-10
        marginal_entropy = calc_marginal_entropy(avg_sample_prob_map).cpu().numpy().astype(np.float32)
        # diversity = calc_sample_diversity(np.concatenate(samples), self.num_samples_cap)
        ged, _, diversity = calc_generalised_energy_distance(np.stack([segmentation.cpu().numpy()]), np.stack(samples), self.num_classes, self.num_samples_cap)

        self.cms.append(torch.cat(cm).detach().cpu().numpy())
        self.ged.append(ged)
        self.diversity.append(diversity)
        self.loglikelihood.append(loglikelihood)
        self.average_pixel_wise_entropy.append(np.mean(marginal_entropy))

        output_dict = {f'sample_{i:d}': sample for i, sample in enumerate(sitk_samples[:self.num_samples_cap])}
        output_dict.update({'marginal_entropy': sitk.GetImageFromArray(marginal_entropy)})
        output_dict.update({'sample_average_prob_map': sitk.GetImageFromArray(avg_sample_prob_map.cpu().numpy().astype(np.float32))})

        del cm, marginal_entropy, sample_prob_maps
        return output_dict
