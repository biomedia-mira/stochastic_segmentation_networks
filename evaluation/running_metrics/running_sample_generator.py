from running_metrics.running_metric import RunningMetric
import torch
import inspect


class RunningSampleGenerator(RunningMetric):
    def __init__(self, classes, require_prob_maps, extra_maps, num_samples, sampler_class, sampler_kwargs,
                 block_size=2):
        super().__init__(classes, require_prob_maps, extra_maps)
        self.num_classes = len(classes)
        self.num_samples = num_samples
        self.sampler_class = sampler_class
        self.sampler_kwargs = sampler_kwargs
        self.block_size = block_size

    def _evaluate(self, segmentation, prediction, prob_maps, mask, extra_maps):
        torch.cuda.empty_cache()
        torch.no_grad()
        extra_maps = extra_maps.copy()
        extra_maps.update({'prob_maps': prob_maps, 'mask': mask})
        params = inspect.signature(self.sampler_class.__dict__['__init__']).parameters.keys()
        kwargs = {key: value for key, value in extra_maps.items() if key in params}
        kwargs.update(self.sampler_kwargs)
        sampler = self.sampler_class(**kwargs)
        sitk_samples = []

        # must be done in blocks due to memory
        for _ in range(self.num_samples // self.block_size):
            sitk_samples_, samples_, prob_map_ = sampler(num_samples=self.block_size)
            sitk_samples += sitk_samples_
            del samples_

        output_dict = {f'sample_{i:d}': sample for i, sample in enumerate(sitk_samples[:self.num_samples])}
        return output_dict
