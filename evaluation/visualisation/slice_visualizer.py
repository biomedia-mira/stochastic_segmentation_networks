import matplotlib.pyplot as plt
import svgutils.transform as sg
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

COLOR_MAP = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0))


# new_size=(230, 230) for brain image
def centered_resize(image, new_size=(230, 230)):
    pad_amount = tuple(max(0, (new_size[i] - image.shape[i])) for i in range(2))
    padding = tuple((pad_amount[i] // 2, pad_amount[i] // 2 + pad_amount[i] % 2) for i in range(2))
    padding = padding + ((0, 0),) if image.ndim == 3 else padding
    new_image = np.pad(image, padding, mode='constant', constant_values=0)

    cropping = tuple(slice(max(0, new_image.shape[i] // 2 - new_size[i] // 2),
                           min(new_image.shape[i], new_image.shape[i] // 2 + new_size[i] // 2 + new_size[i] % 2), 1)
                     for i in range(2))
    return new_image[cropping]


class SliceVisualizer(object):
    def __init__(self,
                 output_path,
                 data_csv_path,
                 image_suffix,
                 extra_image_suffixes=(),
                 overlay_suffixes=(),
                 heat_map_suffixes=(),
                 lower_bound=None,
                 upper_bound=None,
                 new_size=None,
                 color_map=COLOR_MAP,
                 opacity=0.5,
                 number_of_cases_to_plot=None,
                 hide_true_names=False,
                 save_individual_thumbs=True):
        self.output_path = output_path
        self.data_index = pd.read_csv(data_csv_path)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.new_size = new_size
        self.image_suffix = image_suffix
        self.extra_image_suffixes = extra_image_suffixes
        self.overlay_suffixes = overlay_suffixes
        self.heat_map_suffixes = heat_map_suffixes
        self.color_map = np.array(color_map)
        self.opacity = opacity
        self.number_of_cases_to_plot = number_of_cases_to_plot if number_of_cases_to_plot is not None else len(
            self.data_index)
        self.hide_true_names = hide_true_names
        self.save_individual_thumbs = save_individual_thumbs
        self.image_paths = []
        self.all_in_one_image = sg.SVGFigure()
        self.i = 0
        self.j = 0

    def get_slice_numbers(self, image, overlays):
        raise NotImplementedError

    def overlay_to_rgb(self, overlay):
        overlay = overlay.astype(np.int64)
        new_overlay = np.zeros(shape=overlay.shape + (3,))
        for i in range(3):
            new_overlay[..., i] = self.color_map[overlay][..., i]
        return new_overlay

    def image_to_rgb(self, image):
        if self.lower_bound is not None:
            image[image < self.lower_bound] = self.lower_bound
        if self.upper_bound is not None:
            image[image > self.upper_bound] = self.upper_bound
        image = (image - image.min()) / (image.max() - image.min())
        return np.stack((image,) * 3, axis=-1)

    def mix_image_and_overlay(self, image, overlay):
        overlay = overlay.reshape(image.shape)
        new_image = np.copy(image)
        ind = np.sum(overlay, axis=-1) > 0
        new_image[ind] = self.opacity * overlay[ind] + (1 - self.opacity) * image[ind]
        return new_image

    def save_image(self, case_id, image, slice_numbers, suffix='', cmap=None):
        for i, slice_number in enumerate(slice_numbers):
            extension = '_' + str(slice_number) if slice_number != ... else ''
            file_path = os.path.join(self.output_path, str(case_id) + '_slice' + extension + suffix + '.pdf')
            self.image_paths.append(file_path)
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            slice_ = image[slice_number]
            slice_ = centered_resize(slice_, self.new_size) if self.new_size is not None else slice_
            if self.save_individual_thumbs:
                plt.imsave(file_path, slice_, format='pdf', cmap=plt.get_cmap(cmap))

            svg_file_path = file_path.replace('.pdf', '.svg')
            plt.imsave(svg_file_path, slice_, format='svg', cmap=plt.get_cmap(cmap))

            svg_image = sg.fromfile(svg_file_path)
            width = float(svg_image.width.strip('pt'))
            height = float(svg_image.height.strip('pt'))
            plot = svg_image.getroot()
            plot.moveto(width * self.j, height * (self.i + i))
            self.all_in_one_image.append([plot])
            os.remove(svg_file_path)
        self.j += 1

    def __call__(self):
        for i, item in tqdm(self.data_index.iterrows()):
            if i == self.number_of_cases_to_plot:
                break
            self.j = 0
            case_id = 'case_' + str(i) + '/' if self.hide_true_names else item['id']
            image = sitk.GetArrayFromImage(sitk.ReadImage(item[self.image_suffix]))

            overlays = {}
            for overlay_suffix in self.overlay_suffixes:
                overlays[overlay_suffix] = sitk.GetArrayFromImage(sitk.ReadImage(item[overlay_suffix]))

            slice_numbers = self.get_slice_numbers(image, overlays)

            image = self.image_to_rgb(image)
            self.save_image(case_id, image, slice_numbers, '_' + self.image_suffix)

            for suffix in self.extra_image_suffixes:
                extra_image = self.image_to_rgb(sitk.GetArrayFromImage(sitk.ReadImage(item[suffix])))
                self.save_image(case_id, extra_image, slice_numbers, '_' + suffix)

            for suffix, overlay in overlays.items():
                overlay_image = self.mix_image_and_overlay(image, self.overlay_to_rgb(overlay))
                self.save_image(case_id, overlay_image, slice_numbers, '_' + suffix)

            for suffix in self.heat_map_suffixes:
                heat_map = sitk.GetArrayFromImage(sitk.ReadImage(item[suffix]))
                self.save_image(case_id, heat_map, slice_numbers, '_' + suffix, cmap='inferno')

            self.i += 1
        self.all_in_one_image.save(os.path.join(self.output_path, 'thumbs.svg'))


class MostLoadedSliceVisualiser(SliceVisualizer):
    def get_slice_numbers(self, image, overlays):
        segmentation = overlays[list(overlays.keys())[0]]
        slice_numbers = [np.argmax(np.sum(segmentation > 0, axis=(1, 2)))]
        return slice_numbers


class MostLoadedSliceVisualizerPerClass(SliceVisualizer):
    def get_slice_numbers(self, image, overlays):
        segmentation = overlays[list(overlays.keys())[0]]
        slice_numbers = []
        for i in np.unique(segmentation):
            if i != 0:
                slice_numbers.append(np.argmax(np.sum(segmentation == i, axis=(1, 2))))
        return slice_numbers


# useful for 2D data
class DummySliceVisualiser(SliceVisualizer):
    def get_slice_numbers(self, image, overlays):
        return [...]


