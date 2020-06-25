import torch
import os
import numpy as np
from evaluator import Evaluator
from running_metrics.running_sample_generator import RunningSampleGenerator
from running_metrics.samplers import LowRankMultivariateNormalRandomSampler, \
    LowRankMultivariateNormalClassWeightedRangeSampler, CategoricalSampler
from visualisation.slice_visualizer import MostLoadedSliceVisualiser
import argparse

class_names = ['background', 'non-enhancing tumor', 'oedema', 'enhancing tumor']
class_mergers = {'tumor core': {'non-enhancing tumor', 'enhancing tumor'}}
DEVICE = 0


# these are useless, random_sampler is just used to generate noisy samples and deterministic_sampler to
# calculate generalised energy distance without having to change too much code around
def get_samplers_deterministic(num_samples):
    samplers = {'random_sampler_samples_only': {'sampler_class': CategoricalSampler,
                                                'sampler_kwargs': {'device': torch.device(DEVICE), 'seed': None},
                                                'extra_maps': [],
                                                'require_prob_maps': True,
                                                'num_samples': num_samples},
                }
    return samplers


def get_samplers_stochastic(num_samples):
    samplers = {'random_sampler_samples_only': {'sampler_class': LowRankMultivariateNormalRandomSampler,
                                                'sampler_kwargs': {'device': torch.device(DEVICE), 'seed': None},
                                                'extra_maps': ['logit_mean', 'cov_diag', 'cov_factor'],
                                                'require_prob_maps': False,
                                                'num_samples': num_samples}}
    return samplers


def get_class_weigthed_samplers(scale_range=3):
    kwargs = {'device': torch.device(DEVICE), 'from_mean': False, 'seed': 7}
    r = scale_range
    samplers = {}
    for i in range(4):
        kwargs.update({'class_index': i, 'scale_range': np.linspace(-r, r, 2 * r + 1).tolist()})
        samplers.update(
            {f'class_weighted_sampler_c_{i:d}_samples_only': {
                'sampler_class': LowRankMultivariateNormalClassWeightedRangeSampler,
                'sampler_kwargs': kwargs,
                'extra_maps': ['logit_mean', 'cov_diag', 'cov_factor'],
                'require_prob_maps': False,
                'num_samples': 1}})
    return samplers


def evaluate(csv_path, deterministic, detailed=False, make_thumbs=False, num_samples=20):
    running_metrics = {}

    samplers = {}
    if deterministic:
        if detailed:
            samplers = get_samplers_deterministic(num_samples)
    else:
        samplers = get_samplers_stochastic(num_samples)
        if detailed:
            samplers.update(get_class_weigthed_samplers(3))

    for key, sampler in samplers.items():
        num_samples = sampler['num_samples']
        running_metrics.update({key: RunningSampleGenerator(classes=class_names,
                                                            require_prob_maps=sampler['require_prob_maps'],
                                                            extra_maps=sampler['extra_maps'],
                                                            num_samples=num_samples,
                                                            sampler_class=sampler['sampler_class'],
                                                            sampler_kwargs=sampler['sampler_kwargs'],
                                                            block_size=min(2, num_samples))})

    evaluator = Evaluator(class_names,
                          running_metrics,
                          prediction_name='prediction',
                          target_name='seg',
                          mask_name='sampling_mask',
                          prob_map_name='prob_maps')

    running_metrics = evaluator(csv_path)

    for key, sampler in samplers.items():
        if key != 'deterministic_sampler' and make_thumbs:
            make_sample_thumbs(os.path.join(os.path.dirname(csv_path), key + '.csv'), 1, 1)


def make_sample_thumbs(sampling_csv, num_samples, number_of_cases_to_plot=50):
    path = sampling_csv.replace('.csv', '/rgb_slices')
    sample_names = [f'sample_{i:d}' for i in range(num_samples)]
    overlay_suffixes = ['seg', 'prediction'] + sample_names
    heat_map_suffixes = ['marginal_entropy']
    MostLoadedSliceVisualiser(path,
                              sampling_csv,
                              image_suffix='t1ce',
                              overlay_suffixes=overlay_suffixes,
                              heat_map_suffixes=heat_map_suffixes,
                              save_individual_thumbs=False,
                              new_size=(210, 210),
                              number_of_cases_to_plot=number_of_cases_to_plot)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-prediction-csv',
                        required=True,
                        type=str,
                        help='Path to the prediction csv generated during inference')
    parser.add_argument('--is_deterministic',
                        default=False,
                        type=bool,
                        help='Set to true to evaluate the deterministic/benchmark model')
    parser.add_argument('--detailed',
                        default=False,
                        type=bool,
                        help='Set to true to compute extra evaluation and visualisation, takes much more time')
    parser.add_argument('--make-thumbs',
                        default=False,
                        type=bool,
                        help='Set to true to produce and vector image with image sample thumbs')
    parser.add_argument('--num-samples',
                        default=20,
                        type=int,
                        help='Number of samples to generate per case')

    parse_args, unknown = parser.parse_known_args()
    detailed = parse_args.detailed
    make_thumbs = parse_args.make_thumbs
    csv_path = parse_args.path_to_prediction_csv
    deterministic = bool(parse_args.is_deterministic)
    evaluate(csv_path, deterministic, detailed, make_thumbs)
