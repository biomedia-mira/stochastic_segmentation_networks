import numpy as np
from scipy import stats
import math


def np_mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and np version is > 1.9 np.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = np.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                              np.diff(sort, axis=axis) == 0,
                              np.zeros(shape=shape, dtype='bool')],
                             axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[index], counts[index]


def nanstderr(array):
    n = np.sum(np.logical_not(np.isnan(array)))
    return np.nanstd(array) / np.sqrt(n)


def iqr(x):
    if len(x) == 0:
        return np.nan
    q75, q25 = np.nanpercentile(x, [75, 25])
    iqr = q75 - q25
    return iqr


def iqr_v2(x, low=25, high=75, axis=None):
    if len(x) == 0:
        return np.nan
    q_high, q_low = np.nanpercentile(x, [high, low], axis=axis)
    iqr = q_high - q_low
    return iqr, q_high, q_low


def compute_metrics_from_cm(cm):
    """
    Compute a battery of metrics from a confusion matrix
    :param cm: confusion matrix of shape (batch, num_classes, num_classes)
    :return: battery of metrics
    """
    assert isinstance(cm, np.ndarray)
    all_labels = np.einsum('bnd->b', cm).reshape((-1, 1))
    condition_positive = np.einsum('bnd->bn', cm)
    condition_negative = all_labels - condition_positive
    predicted_positive = np.einsum('bnd->bd', cm)
    predicted_negative = all_labels - predicted_positive

    true_positive = np.einsum('bnn->bn', cm)
    false_positive = predicted_positive - true_positive
    false_negative = condition_positive - true_positive
    true_negative = predicted_negative - false_negative

    # True Positive Rate (TPR), Recall, Sensitivity
    tpr = true_positive / condition_positive
    # False Negative Rate (FNR), Miss rate
    fnr = false_negative / condition_positive
    # False Positive Rate (FPR), Fall-out, Probability of false alarm
    fpr = false_positive / condition_negative
    # True Negative Rate (TNR), Specificity, Selectivity
    tnr = true_negative / condition_negative
    # Positive Predicted Value (PPV), Precision
    ppv = true_positive / predicted_positive
    # False Discovery Rate (FDR)
    fdr = false_positive / predicted_positive
    # False Omission Rate (FOR)
    for_ = false_negative / predicted_negative
    # Negative Predicted Value (NPV)
    npv = true_negative / predicted_negative
    # Prevalence
    prevalence = condition_positive / (condition_positive + condition_negative)
    # Accuracy
    accuracy = (true_positive + true_negative) / (condition_positive + condition_negative)
    # F1 Score
    f1 = 2 * (ppv * tpr) / (ppv + tpr)
    # DSC (!= F1 Score)
    dsc = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

    return \
        {'ConditionPositive': condition_positive,
         'ConditionNegative': condition_negative,
         'PredictedPositive': predicted_positive,
         'PredictedNegative': predicted_negative,
         'TruePositive': true_positive,
         'FalsePositive': false_positive,
         'TrueNegative': true_negative,
         'FalseNegative': false_negative,
         'TPR': tpr,
         'FNR': fnr,
         'FPR': fpr,
         'TNR': tnr,
         'PPV': ppv,
         'FDR': fdr,
         'FOR': for_,
         'NPV': npv,
         'Prevalence': prevalence,
         'Accuracy': accuracy,
         'F1Score': f1,
         'DSC': dsc,
         }


def clopper_pearson(k, n, alpha=0.05):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def calculate_roc_auc_standard_error(auc: float, positives: int, negatives: int):
    """
    Calculate the standard error for a ROC AUC using the Hanley and McNeil method:
    The meaning and use of the area under a receiver operating characteristic (ROC) curve.
    https://www.ncbi.nlm.nih.gov/pubmed/7063747
    :param curve_factory:
    :return: The standard error for the ROC AUC
    """
    # auc = curve_factory.get_curve('FPR', 'TPR').area
    # positives = np.array(curve_factory.metrics['ConditionPositive'])[0]
    # negatives = np.array(curve_factory.metrics['ConditionNegative'])[0]
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (positives - 1) * (q1 - auc ** 2) + (negatives - 1) * (q2 - auc ** 2)) / (
            positives * negatives))

    return se


def intra_class_coefficient(x, y, alpha=0.05):
    n = len(x)
    z = np.concatenate((x, y))
    mean = np.mean(z)
    var = (1. / (2. * n)) * (np.sum(np.power(x - mean, 2.)) + np.sum(np.power(y - mean, 2.)))
    r = (1. / (n * var)) * np.sum((x - mean) * (y - mean))

    # j=2
    # var = (2. * math.pow(1 - r, 2.) * math.pow(1 + (n + 1) * r, 2.)) / (n * (n - 1) * j)
    # stddev = math.sqrt(var)
    # lo = max(0, r - 1.96 * stddev)
    # hi = min(1, r + 1.96 * stddev)

    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))

    return r, 0.05, lo, hi


def pearsonr_correlation(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    # # remove outliers
    # ind = reject_outliers(x - y, 4)
    # x = x[ind]
    # y = y[ind]

    x = np.concatenate((x, y))
    y = x

    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi
