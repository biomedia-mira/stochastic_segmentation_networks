import pandas as pd
from tabulate import tabulate
import numpy as np
from evaluation.utils.stats import iqr_v2, nanstderr

DASHES = '-' * 200


class ArrayStatistics(object):
    def __init__(self, array):
        stats = {'finite_els': len(array[np.logical_not(np.isnan(array))]),
                 'mean': np.nanmean(array),
                 'std': np.nanstd(array),
                 'min': np.nanmin(array),
                 'max': np.nanmax(array),
                 'median': np.nanmedian(array)}
        iqr, q_75, q_25 = iqr_v2(array)
        stats.update({'iqr': iqr, 'q_75': q_75, 'q_25': q_25})
        stats.update({'stderr': nanstderr(array)})
        self.stats = stats


def report(dataframes, metrics_to_report=('DSC', 'TPR', 'PPV')):
    string_to_print = ''
    for class_, dataframe in dataframes.items():
        string_to_print += f'Class: {class_:s}\n'
        data = {metric: ArrayStatistics(dataframe[metric]).stats for metric in metrics_to_report}
        df = pd.DataFrame(data).transpose()
        column_order = data[list(data.keys())[0]].keys()
        df = df.loc[:, column_order]
        string_to_print += tabulate(df, headers='keys', tablefmt='psql') + '\n'
    print(string_to_print)


def latex_formatted_report(dataframes, metrics_to_report=('DSC', 'TPR', 'PPV')):
    string_to_print = ''
    for class_ in dataframes:
        for metrics in metrics_to_report:
            stats = ArrayStatistics(dataframes[class_][metrics])
            string_to_print += stats.latex_table_str()
    print(string_to_print)
    print(DASHES + '\n\n')
