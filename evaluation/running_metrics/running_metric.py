import pickle


class RunningMetric(object):
    def __init__(self, classes, require_prob_maps=False, extra_maps=None):
        self.classes = classes.copy()
        self.ids = []
        self.spacings = []
        self.require_prob_maps = require_prob_maps
        self.extra_maps = extra_maps if extra_maps is not None else []

    def evaluate(self, id_, spacing, segmentation, prediction, prob_maps, mask, extra_maps):
        self.ids.append(id_)
        self.spacings.append(spacing)
        return self._evaluate(segmentation, prediction, prob_maps, mask, extra_maps)

    def _evaluate(self, segmentation, prediction, prob_maps, mask, extra_maps):
        raise NotImplementedError

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'rb') as f:
            saved_dict = pickle.load(f)
        for key in self.__dict__:
            self.__dict__[key] = saved_dict[key]
