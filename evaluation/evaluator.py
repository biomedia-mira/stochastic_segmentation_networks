import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from running_metrics.running_metric import RunningMetric


# Needs to be changed to resample at some point
def save_image(dataframe, output_dir, id_, image_name, image):
    path = os.path.join(output_dir, id_ + f'_{image_name:s}.nii.gz')
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    dataframe.loc[id_, image_name] = path
    sitk.WriteImage(image, path)
    return dataframe


class ImageReader(object):
    def __init__(self, segmentation, transform, mask, use_crop):
        self.segmentation = segmentation
        self.transform = transform
        self.use_crop = use_crop

        if mask is not None and self.use_crop:
            if transform is not None:
                mask = self.get_transformed_image(mask)
            mask = sitk.Cast(mask, sitk.sitkInt32)
            f = sitk.LabelShapeStatisticsImageFilter()
            f.Execute(mask)
            bbox = f.GetBoundingBox(1)
            self.crop = (slice(bbox[0], bbox[3]), slice(bbox[1], bbox[4]), slice(bbox[2], bbox[5]))
        else:
            self.crop = None
        self.mask = mask

    def get_transformed_image(self, image):
        if self.transform is not None:
            image = sitk.Resample(image, self.segmentation, self.transform.GetInverse(), sitk.sitkNearestNeighbor)
        return image

    def get_cropped_image(self, image):
        if self.crop is not None:
            image = image[self.crop]
        return image

    def read_image(self, path):
        return self.get_cropped_image(self.get_transformed_image(sitk.ReadImage(path)))


class Evaluator(object):
    def __init__(self, classes,
                 running_metrics,
                 target_name,
                 prediction_name,
                 transform_name=None,
                 mask_name=None,
                 prob_map_name=None,
                 overwrite=False,
                 use_crop=False):
        assert isinstance(running_metrics, dict)
        [isinstance(el, RunningMetric) for el in running_metrics]
        self.classes = classes
        self.running_metrics = running_metrics
        self.target_name = target_name
        self.prediction_name = prediction_name
        self.transform_name = transform_name
        self.mask_name = mask_name
        self.prob_map_name = prob_map_name
        self.overwrite = overwrite
        self.use_crop = use_crop

    @staticmethod
    def get_running_metric_path(prediction_csv, name):
        directory = os.path.dirname(os.path.abspath(prediction_csv))
        return os.path.join(directory, name + '.pickle')

    def get_missing_metrics(self, prediction_csv):
        missing_running_metrics = {}
        for name, running_metric in self.running_metrics.items():
            path = self.get_running_metric_path(prediction_csv, name)
            if os.path.exists(path) and not self.overwrite:
                running_metric.load(path)
            else:
                missing_running_metrics[name] = running_metric
        return missing_running_metrics

    def get_images(self, item, require_prob_maps, extra_maps):

        segmentation = sitk.ReadImage(item[self.target_name])
        transform = sitk.ReadTransform(item[self.transform_name]) if self.transform_name is not None else None
        mask = sitk.ReadImage(item[self.mask_name]) if self.mask_name is not None else None

        reader = ImageReader(segmentation, transform, mask, self.use_crop)

        segmentation = reader.get_cropped_image(segmentation)
        prediction = reader.read_image(item[self.prediction_name])
        mask = reader.read_image(item[self.mask_name]) if self.mask_name is not None else None
        prob_maps = reader.read_image(
            item[self.prob_map_name]) if require_prob_maps and self.prob_map_name is not None else None
        extra_maps = {map_: reader.read_image(item[map_]) for map_ in extra_maps}

        spacing = segmentation.GetSpacing()
        # convert to arrays
        segmentation = sitk.GetArrayFromImage(segmentation).astype(np.uint8)
        prediction = sitk.GetArrayFromImage(prediction).astype(np.uint8)
        prob_maps = sitk.GetArrayFromImage(prob_maps).astype(np.float64) if prob_maps is not None else None
        mask = sitk.GetArrayFromImage(mask).astype(np.uint8) if mask is not None else None
        if mask is not None and prob_maps is not None:
            prob_maps[np.logical_not(mask), 0] = 1.
        extra_maps = {key: sitk.GetArrayFromImage(map_).astype(np.float64) for key, map_ in extra_maps.items()}

        return spacing, segmentation, prediction, prob_maps, mask, extra_maps

    def evaluation_loop(self, prediction_csv, running_metrics, require_prob_maps, extra_maps):
        prediction_dataframe = pd.read_csv(prediction_csv, index_col='id')
        output_dir = os.path.dirname(prediction_csv)
        output_dataframes = {}
        for id_, item in tqdm(prediction_dataframe.iterrows()):
            spacing, segmentation, prediction, prob_maps, mask, extra_maps = self.get_images(item, require_prob_maps, extra_maps)
            for metric_name, running_metric in running_metrics.items():
                output_dict = running_metric.evaluate(id_, spacing, segmentation, prediction, prob_maps, mask, extra_maps)
                if output_dict is not None:
                    if metric_name not in output_dataframes:
                        output_dataframes[metric_name] = prediction_dataframe.copy()
                    for image_name, image in output_dict.items():
                        save_image(output_dataframes[metric_name], os.path.join(output_dir, metric_name), id_, image_name, image)
        for metric_name, dataframe in output_dataframes.items():
            dataframe.to_csv(os.path.join(output_dir, metric_name + '.csv'))

    def __call__(self, prediction_csv):
        running_metrics = self.get_missing_metrics(prediction_csv)
        require_prob_maps = np.logical_or.reduce([el.require_prob_maps for el in running_metrics.values()])
        extra_maps = list(set([map_name for el in running_metrics.values() for map_name in el.extra_maps]))
        if len(running_metrics) > 0:
            self.evaluation_loop(prediction_csv, running_metrics, require_prob_maps, extra_maps)
            for name, running_metric in running_metrics.items():
                running_metric.save(self.get_running_metric_path(prediction_csv, name))
        return self.running_metrics
