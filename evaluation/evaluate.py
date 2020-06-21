import torch
import os
import pandas as pd
import numpy as np
from evaluation.evaluator import Evaluator
from evaluation.running_metrics.running_confusion_matrix import RunningConfusionMatrix
from evaluation.running_metrics.running_probability_distribution import RunningDistributionStatistics
from evaluation.running_metrics.samplers import LowRankMultivariateNormalRandomSampler, \
    LowRankMultivariateNormalClassWeightedRangeSampler, CategoricalSampler, CategoricalDeterministicSampler
from evaluation.metrics.overlap_metrics import OverlapMetrics
from evaluation.visualisation.report import report
from evaluation.visualisation.slice_visualizer import MostLoadedSliceVisualiser
from evaluation.metrics.distribution_statistics import DistributionStatistics
import argparse

class_names = ['background', 'non-enhancing tumor', 'oedema', 'enhancing tumor']
class_mergers = {'tumor core': {'non-enhancing tumor', 'enhancing tumor'}}
# benchmark_csv = 'jobs/BraTS/benchmark/inference/test/predictions/prediction.csv'
# rank_10_mc_20_patch_110_csv = 'jobs/BraTS/rank_10_mc_20_patch_110/inference/test/predictions/prediction.csv'
# rank_10_mc_20_patch_140_csv = 'jobs/BraTS/rank_10_mc_20_patch_140/inference/test/predictions/prediction.csv'
path = '/vol/biomedic/users/mm6818/Projects/variational_hydra/'
benchmark_csv = path + 'jobs/BraTS/benchmark/inference/test/predictions/prediction.csv'
rank_10_mc_20_patch_110_csv = path + 'jobs/BraTS/rank_10_mc_20_patch_110/inference/test/predictions/prediction.csv'
rank_10_mc_20_patch_140_csv = path + 'jobs/BraTS/rank_10_mc_20_patch_140/inference/test/predictions/prediction.csv'
model_csvs = [benchmark_csv, rank_10_mc_20_patch_110_csv, rank_10_mc_20_patch_140_csv]


def add_benchmark_prediction_column_to_csv(benchmark_csv, model_csv):
    benchmark_df = pd.read_csv(benchmark_csv, index_col='id')
    model_df = pd.read_csv(model_csv, index_col='id')
    model_df['benchmark_prediction'] = benchmark_df['prediction']
    model_df.to_csv(model_csv)


# these are useless, random_sampler is just used to generate noisy samples and deterministic_sampler to
# calculate generalised energy distance without having to change too much code around
def get_samplers_deterministic():
    samplers = {'random_sampler': {'sampler_class': CategoricalSampler,
                                   'sampler_kwargs': {'device': torch.device(1), 'seed': None},
                                   'extra_maps': [],
                                   'require_prob_maps': True,
                                   'num_samples': 100},
                'deterministic_sampler': {'sampler_class': CategoricalDeterministicSampler,
                                          'sampler_kwargs': {'device': torch.device(1), 'seed': None},
                                          'extra_maps': [],
                                          'require_prob_maps': True,
                                          'num_samples': 1}
                }
    return samplers


def get_samplers_stochastic():
    samplers = {'random_sampler': {'sampler_class': LowRankMultivariateNormalRandomSampler,
                                   'sampler_kwargs': {'device': torch.device(1), 'seed': None},
                                   'extra_maps': ['logit_mean', 'cov_diag', 'cov_factor'],
                                   'require_prob_maps': False,
                                   'num_samples': 100}}
    return samplers


def get_class_weigthed_samplers(scale_range=3):
    kwargs = {'device': torch.device(1), 'from_mean': False, 'seed': 7}
    r = scale_range
    samplers = {}
    for i in range(4):
        kwargs.update({'class_index': i, 'scale_range': np.linspace(-r, r, 2 * r + 1).tolist()})
        samplers.update(
            {f'class_weighted_sampler_c_{i:d}': {'sampler_class': LowRankMultivariateNormalClassWeightedRangeSampler,
                                                 'sampler_kwargs': kwargs,
                                                 'extra_maps': ['logit_mean', 'cov_diag', 'cov_factor'],
                                                 'require_prob_maps': False,
                                                 'num_samples': 1}})
    return samplers


def evaluate(csv_path, deterministic, detailed=False, make_thumbs=False, num_samples_cap=20):
    running_metrics = {'cm': RunningConfusionMatrix(class_names)}

    samplers = {}
    if deterministic:
        if detailed:
            samplers = get_samplers_deterministic()
    else:
        samplers = get_samplers_stochastic()
        if detailed:
            samplers.update(get_class_weigthed_samplers(3))

    for key, sampler in samplers.items():
        num_samples = sampler['num_samples']
        running_metrics.update({key: RunningDistributionStatistics(classes=class_names,
                                                                   require_prob_maps=sampler['require_prob_maps'],
                                                                   extra_maps=sampler['extra_maps'],
                                                                   num_samples=num_samples,
                                                                   sampler_class=sampler['sampler_class'],
                                                                   sampler_kwargs=sampler['sampler_kwargs'],
                                                                   num_samples_cap=num_samples_cap,
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

    overlap_metrics = OverlapMetrics(running_metrics['cm'])
    overlap_metrics.add_merged_dataframe(class_mergers)
    report(overlap_metrics.dataframes, metrics_to_report=('DSC',))
    for key in samplers.keys():
        dist_stats = DistributionStatistics(running_metrics[key], class_names, class_mergers)
        dist_stats.report(['DSC'])
    return overlap_metrics.dataframes


def make_sample_thumbs(sampling_csv, num_samples, number_of_cases_to_plot=50):
    add_benchmark_prediction_column_to_csv(benchmark_csv, sampling_csv)
    path = sampling_csv.replace('.csv', '/rgb_slices')
    sample_names = [f'sample_{i:d}' for i in range(num_samples)]
    overlay_suffixes = ['seg', 'benchmark_prediction', 'prediction'] + sample_names
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
    parser.add_argument('--detailed',
                        default=False,
                        type=bool,
                        help='Set to true to compute extra evaluation and visualisation, takes much more time')
    parser.add_argument('--make-thumbs',
                        default=False,
                        type=bool,
                        help='Set to true to produce and vector image with image sample thumbs')

    parse_args, unknown = parser.parse_known_args()
    detailed = parse_args.detailed
    make_thumbs = parse_args.make_thumbs

    overlap_metrics_dataframes = {}
    for csv_path in model_csvs:
        print(csv_path)
        deterministic = True if csv_path == benchmark_csv else False
        overlap_metrics_dataframes[csv_path] = evaluate(csv_path, deterministic, detailed, make_thumbs)
