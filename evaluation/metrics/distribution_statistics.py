from running_metrics.running_probability_distribution import RunningDistributionStatistics
import numpy as np
from utils.stats import compute_metrics_from_cm, iqr_v2
from metrics.overlap_metrics import merge_cm_classes
from tabulate import tabulate
import matplotlib.pyplot as plt
from utils.stats import nanstderr


def add_volume_metrics(metrics, voxel_spacing):
    metrics['True Volume (ml)'] = metrics['ConditionPositive'] * voxel_spacing
    metrics['Predicted Volume (ml)'] = metrics['PredictedPositive'] * voxel_spacing
    metrics['Absolute Error (mL)'] = np.abs(metrics['True Volume (ml)'] - metrics['Predicted Volume (ml)'])
    relative_error = 100 * np.abs(metrics['Absolute Error (mL)'] / (metrics['True Volume (ml)']))
    metrics['Relative Error (%)'] = relative_error
    return metrics


class DistributionStatistics(object):
    def __init__(self, runnning_dist_stats: RunningDistributionStatistics, class_names, class_mergers=None):
        self.class_mergers = class_mergers
        self.cms = np.array(runnning_dist_stats.cms)
        class_names[0] = 'lesion (any)'
        self.class_names = class_names
        self.voxel_volume = np.prod([.1, .1, .1])
        self.per_class_metrics = self._compute_metrics_from_cms(class_mergers)
        self._compute_metrics_from_cms(class_mergers)

        self.overall_metrics = {'loglikelihood': np.array(runnning_dist_stats.loglikelihood),
                                'diversity': np.array(runnning_dist_stats.diversity),
                                'ged': np.array(runnning_dist_stats.ged),
                                'average_pixel_wise_entropy': np.array(runnning_dist_stats.average_pixel_wise_entropy)}

    def _compute_metrics_from_cms(self, class_mergers):
        per_class_metrics = {key: {} for key in self.class_names}
        batch_shape = self.cms.shape[:2]
        num_classes = self.cms.shape[3]

        cm = self.cms.reshape((-1, num_classes, num_classes))
        metrics = compute_metrics_from_cm(cm)
        metrics = add_volume_metrics(metrics, self.voxel_volume)
        for i, class_name in enumerate(self.class_names):
            for metric, value in metrics.items():
                per_class_metrics[class_name][metric] = value[:, i].reshape(batch_shape)

        foreground_merge = {'lesion (any)': self.class_names[1:]}
        if class_mergers is None:
            class_mergers = foreground_merge
        else:
            class_mergers.update(foreground_merge)
        for class_name in class_mergers.keys():
            per_class_metrics[class_name] = {}

        for class_name, foreground_classes in class_mergers.items():
            fi = [self.class_names.index(el) for el in foreground_classes]
            metrics = compute_metrics_from_cm(merge_cm_classes(cm, fi))
            metrics = add_volume_metrics(metrics, self.voxel_volume)
            for metric, value in metrics.items():
                per_class_metrics[class_name][metric] = value[:, 1].reshape(batch_shape)
        return per_class_metrics

    def report(self, metrics_to_report=('DSC', 'Predicted Volume (ml)')):
        functions = {'mean': np.nanmean,
                     'std': np.nanstd,
                     'median': np.nanmedian,
                     'min': np.nanmin,
                     'max': np.nanmax}

        def get_metric_from_array(array):
            metrics = {key: func(array, axis=-1) for key, func in functions.items()}
            metrics['iqr25_75'], metrics['q_75'], metrics['q_25'] = iqr_v2(array, 25, 75, axis=-1)
            metrics['iqr05_95'], metrics['q_95'], metrics['q_05'] = iqr_v2(array, 5, 95, axis=-1)
            return metrics

        table = []
        for key, value in self.overall_metrics.items():
            table.append([key] + [f'{np.nanmean(value):.4f} ± {nanstderr(value):.4f}'])
        print(tabulate(table, headers=self.overall_metrics.keys(), tablefmt="plain"))

        headers = None
        for metric_to_report in metrics_to_report:
            table = []
            for class_name in self.per_class_metrics.keys():
                array = self.per_class_metrics[class_name][metric_to_report]
                metrics = get_metric_from_array(array)
                table.append(
                    [class_name] + [f'{np.nanmean(value):.4f} ± {np.nanstd(value):.4f}' for value in metrics.values()])
                if headers is None:
                    headers = ['class_name'] + list(metrics.keys())
            avg = np.mean(
                np.array([self.per_class_metrics[class_name][metric_to_report] for class_name in self.class_names[1:]]),
                axis=0)
            metrics = get_metric_from_array(avg)
            table.append(['avg'] + [f'{np.nanmean(value):.4f} ± {np.nanstd(value):.4f}' for value in metrics.values()])
            print(metric_to_report)
            print(tabulate(table, headers=headers, tablefmt="plain"))

    def sample_distribution_plot(self, metric_name, overlap_metrics_dataframes=None):
        per_class_data = {}
        per_class_benchmark = {}
        for class_, metrics in self.per_class_metrics.items():
            data = metrics[metric_name]
            order = np.argsort(np.median(data, axis=-1))
            per_class_data[class_] = data[order]
            if overlap_metrics_dataframes is not None:
                per_class_benchmark[class_] = overlap_metrics_dataframes[class_][metric_name][order]

        all_data = np.array(
            [per_class_data[class_] for class_ in per_class_data.keys() if class_ not in self.class_mergers])
        avg_all_data = np.nanmean(all_data, axis=0)
        order = np.argsort(np.median(avg_all_data, axis=-1))
        per_class_data['avg'] = avg_all_data[order]
        if overlap_metrics_dataframes is not None:
            benchmark = np.array([per_class_benchmark[class_] for class_ in per_class_benchmark.keys() if
                                  class_ not in self.class_mergers])
            benchmark_avg = np.nanmean(benchmark, axis=0)
            per_class_benchmark['avg'] = np.array(benchmark_avg[order])

        for c, (class_, metrics) in enumerate(per_class_data.items()):
            fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
            boxplot = ax.boxplot(per_class_data[class_].transpose(),
                                 boxprops=dict(facecolor='#0496FF', color='#0496FF'),
                                 medianprops=dict(color='black'),
                                 flierprops=dict(color='#0496FF', markeredgecolor='#0496FF'),
                                 patch_artist=True)
            ax.grid(axis='both', which='both', zorder=-5)
            # COLOR_SCHEME = ['#800080', '#FF0000', '#26CC14', '#0496FF', '#F4B400']

            if overlap_metrics_dataframes is not None:
                benchmark = per_class_benchmark[class_]
                x = np.array(list(range(1, len(benchmark) + 1)), dtype=np.float32)
                ax.set_ylabel('DSC', fontweight='bold', fontsize=15)
                ax.scatter(x, benchmark, marker='x', color='black', zorder=10)
                quality = np.mean(per_class_data[class_] > np.expand_dims(benchmark, axis=-1), axis=-1)
                ax.bar(x, quality, color='#ffd604', zorder=-1, edgecolor='#9e9100', align='center', width=1.)
                plt.plot([0, x[-1] + 1], [np.nanmean(quality)] * 2, linestyle='--', color='black')

            ax.set_xticks([])
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(12)
                tick.label1.set_fontweight('bold')
            ax.set_ylim((0, 1))
            fig.show()

