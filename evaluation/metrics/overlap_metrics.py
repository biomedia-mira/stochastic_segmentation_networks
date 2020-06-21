import os
import numpy as np
import pandas as pd
from running_metrics.running_confusion_matrix import RunningConfusionMatrix
from utils.stats import compute_metrics_from_cm


def compress_running_cm(cm):
    return np.sum(cm, axis=-1)


def calc_relative_error(x):
    e = 100 * np.abs(x['Absolute Error (mL)'] / (x['True Volume (ml)']))
    return e[np.isfinite(e)]


def update_dataframe_with_volume_metrics(dataframe):
    dataframe['True Volume (ml)'] = dataframe['ConditionPositive'] * dataframe['voxel_volume_ml']
    dataframe['Predicted Volume (ml)'] = dataframe['PredictedPositive'] * dataframe['voxel_volume_ml']
    if 'SumOfProbability' in dataframe:
        dataframe['Predicted Soft Volume (ml)'] = dataframe['SumOfProbability'] * dataframe['voxel_volume_ml']
    dataframe['Absolute Error (mL)'] = np.abs(dataframe['True Volume (ml)'] - dataframe['Predicted Volume (ml)'])
    dataframe['Relative Error (%)'] = calc_relative_error(dataframe)
    return dataframe


def merge_cm_classes(cm, foreground_classes_indices):
    assert isinstance(cm, np.ndarray)

    new_cm = np.zeros(shape=(cm.shape[0], 2, 2))

    # foreground indices
    fi = foreground_classes_indices
    # background indices
    bi = list(set(range(cm.shape[1])) - set(foreground_classes_indices))

    true_positive = np.sum(cm[:, fi][:, :, fi], axis=(1, 2))
    true_negative = np.sum(cm[:, bi][:, :, bi], axis=(1, 2))
    false_negative = np.sum(cm[:, fi][:, :, bi], axis=(1, 2))
    false_positive = np.sum(cm[:, bi][:, :, fi], axis=(1, 2))

    new_cm[:, 1, 1] = true_positive
    new_cm[:, 0, 0] = true_negative
    new_cm[:, 1, 0] = false_negative
    new_cm[:, 0, 1] = false_positive
    return new_cm


def compute_voxel_volume_ml(spacing):
    return np.prod(spacing) / 1000.


def cm_metrics_to_dataframe(ids, spacings, cm_metrics, class_index):
    voxel_volume_ml = [compute_voxel_volume_ml(spacing) for spacing in spacings]
    dataframe = pd.DataFrame({'id': ids,
                              'spacing_mm': spacings,
                              'voxel_volume_ml': voxel_volume_ml})
    for key in cm_metrics:
        dataframe[key] = cm_metrics[key][:, class_index]
    return dataframe


class OverlapMetrics(object):
    def __init__(self, running_confusion_matrix):
        assert isinstance(running_confusion_matrix, RunningConfusionMatrix)
        self.cm = np.array(running_confusion_matrix.cm.copy())
        self.classes = running_confusion_matrix.classes.copy()
        self.classes[0] = 'lesion (any)'
        self.ids = running_confusion_matrix.ids.copy()
        self.spacings = running_confusion_matrix.spacings.copy()
        self.dataframes = self.get_class_dataframes_from_cm()
        self.filtered_dataframes = self.dataframes

    @staticmethod
    def get_class_dataframe(ids, spacings, metrics, class_index):
        dataframe = cm_metrics_to_dataframe(ids, spacings, metrics, class_index)
        dataframe = update_dataframe_with_volume_metrics(dataframe)
        return dataframe

    def add_merged_dataframe(self, class_mergers=None):
        if class_mergers is not None:
            for class_, foreground_classes in class_mergers.items():
                fi = [self.classes.index(el) for el in foreground_classes]
                metrics = compute_metrics_from_cm(merge_cm_classes(self.cm, fi))
                self.dataframes[class_] = self.get_class_dataframe(self.ids, self.spacings, metrics, 1)

    def get_class_dataframes_from_cm(self):
        dataframes = dict.fromkeys(self.classes)
        metrics = compute_metrics_from_cm(self.cm)
        foreground_metrics = compute_metrics_from_cm(merge_cm_classes(self.cm, list(range(1, len(self.classes)))))

        for i, class_ in enumerate(self.classes):
            if i == 0:
                dataframes[class_] = self.get_class_dataframe(self.ids, self.spacings, foreground_metrics, 1)
            else:
                dataframes[class_] = self.get_class_dataframe(self.ids, self.spacings, metrics, i)

        return dataframes

    def filter_by(self, variable, lambda_condition):
        self.filtered_dataframes = {class_: dataframe[lambda_condition(dataframe[variable])] for class_, dataframe in
                                    self.filtered_dataframes.items()}

    def add_variable(self, variable_name, lambda_expression):
        for class_, dataframe in self.dataframes.items():
            self.dataframes[class_][variable_name] = lambda_expression(dataframe)

    def remove_all_filters(self):
        self.filtered_dataframes = self.dataframes

    def save_dataframes(self, directory):
        for class_name, dataframe in self.dataframes.items():
            dataframe.to_csv(os.path.join(directory, class_name + '_overlap_metrics.csv'), index=False)
