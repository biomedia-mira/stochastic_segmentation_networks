import numpy as np


# All transformations can't alter state after __init__ because of dataset class when num_workers > 0


class Transformation(object):
    """
    Abstract class for case transformation.
    __call__: When called a CaseTransformation should return an image and target with the same shape as the input.
    """

    def __call__(self, image, target, sampling_mask):
        raise NotImplementedError


class IntensityWindowNormalization(Transformation):
    def __init__(self, lower_bound, upper_bound, map_upper_bound_to_lower_bound=False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.map_upper_bound_to_lower_bound = map_upper_bound_to_lower_bound

    def __call__(self, image, target, sampling_mask):

        sampling_mask = np.logical_and(sampling_mask,
                                       np.squeeze(np.logical_and(image >= self.lower_bound, image <= self.upper_bound)))

        image[image <= self.lower_bound] = self.lower_bound
        if self.map_upper_bound_to_lower_bound:
            image[image >= self.upper_bound] = self.lower_bound
        else:
            image[image >= self.upper_bound] = self.upper_bound

        # normalize
        image = (2. * image - self.lower_bound - self.upper_bound) / (self.upper_bound - self.lower_bound)

        return image, target, sampling_mask


class MaskImageUsingSamplingMask(Transformation):
    def __init__(self, outside_value):
        self.outside_value = outside_value

    def __call__(self, image, target, sampling_mask):
        image[np.broadcast_to(np.logical_not(sampling_mask), shape=image.shape)] = self.outside_value
        return image, target, sampling_mask


class BrainMaskFromChannel(Transformation):
    def __init__(self, channel_idx=0):
        self.channel_idx = channel_idx

    def __call__(self, image, target, sampling_mask):
        sampling_mask = np.logical_and(sampling_mask, image[self.channel_idx] > 0)
        return image, target, sampling_mask


class PerChannelZScoreNormalisation(Transformation):
    def __init__(self, cutoff_percentiles=(5., 95.), cutoff_below_mean=True):
        self.cutoff_percentiles = cutoff_percentiles
        self.cutoff_below_mean = cutoff_below_mean

    def __call__(self, image, target, sampling_mask):
        for i, channel in enumerate(image):
            low, high = np.percentile(channel[sampling_mask], self.cutoff_percentiles)
            norm_mask = np.logical_and(sampling_mask, np.logical_and(channel > low, channel < high))
            if self.cutoff_below_mean:
                norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))

            masked_channel = channel[norm_mask]
            normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
            image[i] = normalised_channel

        return image, target, sampling_mask
