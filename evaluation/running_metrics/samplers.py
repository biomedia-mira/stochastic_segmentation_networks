import torch
import torch.distributions as td
from torch.distributions.lowrank_multivariate_normal import _standard_normal, _batch_mv
import SimpleITK as sitk
import numpy as np


def cast_to_tensor(array, device, dtype=torch.float64):
    return torch.tensor(array.transpose((-1,) + tuple(range(array.ndim - 1))), dtype=dtype, device=device,
                        requires_grad=False)


def tensors_to_sitk_images(samples):
    return [sitk.Cast(sitk.GetImageFromArray(sample.cpu().numpy().astype(float)), sitk.sitkUInt8) for sample in samples]


class Sampler(object):
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, num_samples):
        raise NotImplementedError


# this is to show what happens if you sample from independent categoricals, no one does this in practise for good reason
class CategoricalSampler(Sampler):
    def __init__(self, prob_maps, device, seed=None):
        super().__init__(seed)
        self.prob_maps = torch.tensor(prob_maps, dtype=torch.float32, device=device, requires_grad=False)
        self.dist = td.Independent(td.Categorical(self.prob_maps), len(prob_maps.shape) - 1)

    def __call__(self, num_samples):
        samples = self.dist.sample([num_samples])
        prob_maps = torch.stack([self.prob_maps.permute((-1,) + tuple(range(self.prob_maps.dim() - 1)))] * len(samples))
        return tensors_to_sitk_images(samples), samples, prob_maps


# not actually a sampler it's just the argmax, just makes it easier to compute D_GED^2 without changing too much code
class CategoricalDeterministicSampler(Sampler):
    def __init__(self, prob_maps, device, seed=None):
        super().__init__(seed)
        self.prob_maps = torch.tensor(prob_maps, dtype=torch.float32, device=device, requires_grad=False)

    def __call__(self, num_samples):
        samples = torch.stack([torch.argmax(self.prob_maps, axis=-1)] * num_samples)
        prob_maps = torch.stack([self.prob_maps.permute((-1,) + tuple(range(self.prob_maps.dim() - 1)))] * len(samples))
        return tensors_to_sitk_images(samples), samples, prob_maps


class LowRankMultivariateNormalRandomSampler(Sampler):
    def __init__(self, logit_mean, cov_diag, cov_factor, device, mask, seed=None):
        super().__init__(seed)
        self.dist, self.shape, self.rank = self.build_distribution(logit_mean, cov_diag, cov_factor, device, mask)

    @staticmethod
    def build_distribution(logit_mean, cov_diag, cov_factor, device, mask):
        logit_mean = cast_to_tensor(logit_mean, device)
        cov_diag = cast_to_tensor(cov_diag, device)
        cov_factor = cast_to_tensor(cov_factor, device)
        if mask is not None:
            mask = torch.tensor(mask, device=device, requires_grad=False)
            logit_mean[0, ~mask.type(torch.bool)] = 100
            cov_factor = cov_factor * mask.unsqueeze(0)
        shape = logit_mean.shape
        num_classes = shape[0]
        rank = int(cov_factor.shape[0] / num_classes)
        logit_mean = logit_mean.view(-1)
        cov_diag = cov_diag.view(-1)
        cov_factor = cov_factor.view((rank, -1)).transpose(1, 0)
        epsilon = 1e-3
        dist = td.LowRankMultivariateNormal(loc=logit_mean, cov_factor=cov_factor, cov_diag=cov_diag + epsilon)
        return dist, shape, rank

    def __call__(self, num_samples):
        logit_samples = self.dist.sample([num_samples])
        logit_samples = logit_samples.view((num_samples,) + self.shape)
        prob_maps = torch.softmax(logit_samples, dim=1)
        samples = torch.argmax(logit_samples, dim=1)
        return tensors_to_sitk_images(samples), samples, prob_maps


class LowRankMultivariateNormalTemperatureScaledRandomSampler(LowRankMultivariateNormalRandomSampler):
    def __init__(self, logit_mean, cov_diag, cov_factor, device, mask, temperature, seed=None):
        super().__init__(logit_mean, cov_diag, cov_factor, device, mask, seed)
        self.temperature = temperature

    def normal_dist(self, shape):
        dtype = self.dist.loc.dtype
        device = self.dist.loc.device
        return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
                            self.temperature * torch.ones(shape, dtype=dtype, device=device))

    def __call__(self, num_samples):
        shape = self.dist._extended_shape([num_samples])
        w_shape = shape[:-1] + self.dist.cov_factor.shape[-1:]
        eps_w = self.normal_dist(w_shape)
        eps_d = self.normal_dist(shape)
        factor_direction = _batch_mv(self.dist._unbroadcasted_cov_factor, eps_w)
        diag_direction = self.dist._unbroadcasted_cov_diag.sqrt() * eps_d
        logit_samples = self.dist.loc + factor_direction + diag_direction
        logit_samples = logit_samples.view((num_samples,) + self.shape)
        prob_maps = torch.softmax(logit_samples, dim=1)
        samples = torch.argmax(logit_samples, dim=1)
        return tensors_to_sitk_images(samples), samples, prob_maps


class LowRankMultivariateNormalWeightedSampler(LowRankMultivariateNormalRandomSampler):
    def __init__(self, logit_mean, cov_diag, cov_factor, device, mask, seed=None):
        super().__init__(logit_mean, cov_diag, cov_factor, device, mask, seed)
        self.eps_W = None
        self.eps_D = None
        self.device = device

    def iter_random_state(self, num_samples):
        shape = self.dist._extended_shape([num_samples])
        W_shape = shape[:-1] + self.dist.cov_factor.shape[-1:]
        self.eps_W = _standard_normal(W_shape, dtype=self.dist.loc.dtype, device=self.dist.loc.device)
        self.eps_D = _standard_normal(shape, dtype=self.dist.loc.dtype, device=self.dist.loc.device)

    def get_weighted_samples(self,
                             temperature = 1.,
                             rank_weights: torch.Tensor = None,
                             class_weights: torch.Tensor = None):
        rank_weights = torch.ones(self.rank, device=self.device) if rank_weights is None else \
            torch.tensor(rank_weights, device=self.device)
        class_weights = torch.ones(self.num_classes, device=self.device) if class_weights is None else \
            torch.tensor(class_weights, device=self.device)

        factor_direction = _batch_mv(self.dist._unbroadcasted_cov_factor, self.eps_W * rank_weights)
        diag_direction = self.dist._unbroadcasted_cov_diag.sqrt() * self.eps_D

        spatial_size = int(self.dist.event_shape[-1] // self.num_classes)
        class_weights = torch.repeat_interleave(class_weights, spatial_size)
        logit_samples = self.dist.loc + class_weights * (factor_direction + diag_direction)

        logit_samples = logit_samples.view((-1,) + self.shape)
        prob_maps = torch.softmax(logit_samples, dim=1)
        samples = torch.argmax(logit_samples, dim=1)
        return tensors_to_sitk_images(samples), samples, prob_maps

    def __call__(self, num_samples):
        raise NotImplementedError


class LowRankMultivariateNormalClassWeightedRangeSampler(LowRankMultivariateNormalWeightedSampler):
    def __init__(self, logit_mean, cov_diag, cov_factor, device, mask,
                 class_index, scale_range=(-1, 0, 1), from_mean=False, seed=7):
        super().__init__(logit_mean, cov_diag, cov_factor, device, mask, seed)
        self.class_index = class_index
        self.scale_range = scale_range
        self.from_mean = from_mean
        self.num_classes = self.shape[0]

    def __call__(self, num_samples):
        prob_maps = []
        self.iter_random_state(num_samples)
        shape = (len(self.scale_range), self.num_classes)
        scale_range = np.zeros(shape) if self.from_mean else np.ones(shape)
        scale_range[:, self.class_index] = self.scale_range
        for scale in scale_range:
            _, _, _prob_maps = self.get_weighted_samples(class_weights=scale)
            prob_maps.append(_prob_maps)
        prob_maps = torch.stack(prob_maps)
        samples = torch.argmax(prob_maps, dim=2)
        sitk_samples = [sitk.GetImageFromArray(sample.cpu().numpy().astype(np.uint8))
                        for sample in samples.permute(1, 2, 3, 4, 0)]
        return sitk_samples, samples.view((-1,) + samples.shape[2:]), prob_maps.view((-1,) + prob_maps.shape[2:])
