import torch
import os
import torch.distributions as td
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn as nn
import pickle
import copy
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal, lazy_property

CMAP = 'cividis'


# visualize
def show_cov_matrix(cov):
    cov = cov.detach().numpy()
    lim = np.max(np.abs(cov))
    plt.imshow(cov, cmap='seismic', clim=(-lim, lim))
    cbar = plt.colorbar()
    cbar.minorticks_on()
    plt.show()


def show_images(samples, dim, output_path='samples.pdf'):
    images = samples.unsqueeze(-1).repeat((1, 1, dim)).detach().numpy()
    fig, axarr = plt.subplots(1, len(samples), squeeze=True)
    for j in range(len(samples)):
        axarr[j].imshow(images[j], cmap=CMAP)
        axarr[j].set_xticklabels([])
        axarr[j].set_yticklabels([])
    fig.savefig(output_path, bbox_inches='tight')
    # fig.show()


def calculate_loglikelihood(distribution, dim, num_samples=1000):
    target = get_on_off_binary_target(dim, 0)
    logit_sample = distribution.rsample([num_samples, 2])
    target = expand_target(target, num_samples)
    prob = target * torch.sigmoid(logit_sample) + (1 - target) * (1 - torch.sigmoid(logit_sample))
    loglikelihood = torch.mean(torch.logsumexp(torch.sum(torch.log(prob), dim=-1), dim=0) - math.log(num_samples))
    return float(loglikelihood.detach().numpy())


def evaluate_distribution(dist, dim, num_samples=10, name='dist_eval.pdf'):
    width = dim // 3
    print(f'Model has loglikelihood {calculate_loglikelihood(dist, dim):.5f}')
    mean = dist.mean.unsqueeze(-1).repeat(1, 1, width).detach().numpy()
    covariance_matrix = dist.covariance_matrix.detach().numpy()
    lim = np.max(np.abs(covariance_matrix))

    fig = plt.figure(figsize=(12, 4), constrained_layout=False)

    ncols = 3 * (len(mean) + 1) + num_samples // 2
    width_ratios = np.array([1] * ncols)
    width_ratios[3 * (len(mean))] = dim / float(width)
    width_ratios[3 * (len(mean)) + 1] = dim / float(width)

    gs = fig.add_gridspec(nrows=2, ncols=ncols, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.05)

    for i, m in enumerate(mean):
        ax = fig.add_subplot(gs[:, (3 * i):(3 * i + 2)])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        map = ax.imshow(m, cmap=CMAP)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(mappable=map, cax=cax)
        # cbar.ax.tick_params(labelsize=5)
        cbar.minorticks_on()

    ax = fig.add_subplot(gs[:, (3 * len(mean)):(3 * len(mean) + 2)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    map = ax.imshow(covariance_matrix, cmap='seismic', clim=(-lim, lim))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable=map, cax=cax)
    # cbar.ax.tick_params(labelsize=5)
    cbar.minorticks_on()

    logit_samples = dist.rsample([num_samples])
    samples = torch.round(torch.sigmoid(logit_samples))
    for i in range(2):
        for j in range(len(samples) // 2):
            ax = fig.add_subplot(gs[i, j + 3 * (len(mean) + 1)])
            sample = samples[i + j].unsqueeze(-1).repeat((1, width)).detach().numpy()
            ax.imshow(sample, cmap=CMAP)
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    # fig.show()
    fig.savefig(name)


def get_prob(logit_sample, target, num_classes):
    if num_classes == 2:
        prob = target * torch.sigmoid(logit_sample) + (1 - target) * (1 - torch.sigmoid(logit_sample))
    else:
        raise NotImplementedError
    return prob


# toy problems
def get_on_off_binary_target(dim, noise_level=0.):
    target = torch.zeros((2, dim), dtype=torch.int64)
    target[:, :dim // 3] = 1
    target[0, dim // 3: 2 * dim // 3] = 0
    target[1, dim // 3: 2 * dim // 3] = 1
    noise = torch.rand((2, dim)) > 1 - noise_level
    target[noise] = torch.logical_not(target[noise]).type(torch.int64)
    return target


def expand_target(target, num_mc_samples):
    return target.repeat((num_mc_samples, 1, 1))


def get_on_off_ideal_solution(dim, alpha=1000):
    mean = torch.zeros(dim)
    mean[:dim // 3] = alpha
    mean[dim // 3:2 * dim // 3] = 0
    mean[2 * dim // 3:] = -alpha
    cov_matrix = torch.zeros((dim, dim))
    cov_matrix = cov_matrix + torch.eye(dim) * 1e-3
    cov_matrix[dim // 3:2 * dim // 3, dim // 3:2 * dim // 3] = 2
    return td.MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)


def get_slide_bar_binary_target(dim, noise_level=0):
    slides = list(range(dim // 4, 3 * dim // 4))
    target = torch.zeros((len(slides), dim), dtype=torch.int64)
    target[:, 3 * dim // 4:] = 1
    for i, s in enumerate(slides):
        target[i, s:3 * dim // 4] = 1
    noise = torch.rand((len(slides), dim)) > 1 - noise_level
    target[noise] = torch.logical_not(target[noise]).type(torch.int64)
    return target


# lightweight port form the td.distribution: does not crash inverting matrices unnecessarily
class LowRankMultivariateNormal(object):
    def __init__(self, loc, cov_factor, cov_diag):
        self.loc = loc
        self.cov_diag = cov_diag
        self.cov_factor = cov_factor
        self._batch_shape = self.loc.shape[:-1]
        self._event_shape = self.loc.shape[-1:]

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def covariance_matrix(self):
        covariance_matrix = (torch.matmul(self.cov_factor,
                                          self.cov_factor.transpose(-1, -2))
                             + torch.diag_embed(self.cov_diag))
        return covariance_matrix.expand(self._batch_shape + self._event_shape +
                                        self._event_shape)

    def rsample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self._batch_shape + self._event_shape
        W_shape = shape[:-1] + self.cov_factor.shape[-1:]
        eps_W = _standard_normal(W_shape, dtype=self.loc.dtype, device=self.loc.device)
        eps_D = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self.cov_factor, eps_W) + self.cov_diag.sqrt() * eps_D


# Diagonal rank gaussian
class DiagonalModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float64), requires_grad=True)
        self.log_cov_diag = torch.nn.Parameter(-torch.ones(dim, dtype=torch.float64), requires_grad=True)

    def get_dist(self):
        return td.MultivariateNormal(loc=self.mean, covariance_matrix=torch.diag(self.log_cov_diag.exp()))


# Low rank gaussian
class LowRankModel(nn.Module):
    def __init__(self, dim, rank=2):
        super().__init__()
        self.rank = rank
        self.mean = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float64), requires_grad=True)
        self.log_cov_diag = torch.nn.Parameter(-torch.ones(dim, dtype=torch.float64), requires_grad=True)
        self.log_cov_factor_scale = torch.nn.Parameter(-torch.ones(dim, rank, dtype=torch.float64), requires_grad=True)
        self.cov_factor_direction = torch.nn.Parameter(torch.rand(dim, rank, dtype=torch.float64) - 0.5,
                                                       requires_grad=True)

    def get_dist(self):
        cov_factor = self.log_cov_factor_scale.exp() * (
                    self.cov_factor_direction / torch.norm(self.cov_factor_direction, dim=0, keepdim=True))
        cov_factor = cov_factor.view(-1, self.rank)
        # lim = 3. / self.rank
        # cov_factor = cov_factor.clamp(min=-lim, max=lim)
        #
        return LowRankMultivariateNormal(loc=self.mean, cov_factor=cov_factor, cov_diag=self.log_cov_diag.exp())


def run_toy_problem(model_type, dim=21, get_target_fn=get_on_off_binary_target):
    if model_type == 'low_rank':
        p_gen = LowRankModel(dim, rank=2)
    elif model_type == 'diagonal':
        p_gen = DiagonalModel(dim)
    else:
        raise NotImplementedError

    num_epochs = 5000
    num_pre_epochs = 5000
    num_mc_samples = 200
    noise_level = 0.0
    num_classes = 2
    checkpoint = None
    # Train by Monte Carlo integration
    optimizer_mean = torch.optim.Adam(lr=1e-3, params=[p_gen.mean])
    optimizer_all = torch.optim.Adam(lr=1e-3, params=p_gen.parameters())
    for epoch in range(num_pre_epochs + num_epochs):
        optimizer = optimizer_mean if epoch < num_pre_epochs else optimizer_all
        optimizer.zero_grad()
        p = p_gen.get_dist()
        target = get_target_fn(dim, noise_level)
        data_size = target.shape[0]
        logit_sample = p.rsample([num_mc_samples, data_size])
        target = expand_target(target, num_mc_samples)
        prob = get_prob(logit_sample, target, num_classes)
        loglikelihood = torch.mean(
            torch.logsumexp(torch.sum(torch.log(prob), dim=-1), dim=0) - math.log(num_mc_samples))
        loss = -loglikelihood

        if epoch % 500 == 0:
            checkpoint = copy.deepcopy(p_gen)
            print(f'epoch:{epoch:d}: *(noisy) LogLikelihood: {loglikelihood.detach().numpy():.6f}')
        loss.backward()
        optimizer.step()
        if torch.isnan(loss):
            return checkpoint

    return p_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite',
                        default=False,
                        type=bool,
                        help='Whether to overwrite previous run.')
    parse_args, unknown = parser.parse_known_args()
    overwrite = parse_args.overwrite
    dim = 21
    get_target_fn = get_on_off_binary_target

    output_dir = 'jobs/toy_problem'
    os.makedirs(output_dir, exist_ok=True)

    target_sample = get_target_fn(dim, noise_level=0)
    show_images(target_sample, dim, output_path=os.path.join(output_dir, 'target_samples.pdf'))

    for model_type in ['low_rank', 'diagonal']:
        dist_path = os.path.join(output_dir, model_type + '_dist.model')
        if overwrite or not os.path.exists(dist_path):
            p_gen = run_toy_problem(model_type, dim, get_target_fn)
            with open(dist_path, 'wb') as f:
                pickle.dump(p_gen, f)
        else:
            with open(dist_path, 'rb') as f:
                p_gen = pickle.load(f)
        evaluate_distribution(p_gen.get_dist(), dim, name=os.path.join(output_dir, model_type + '.pdf'), num_samples=14)
