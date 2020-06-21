from running_metrics.running_metric import RunningMetric
import numpy as np


class RunningConfusionMatrix(RunningMetric):
    def __init__(self, classes):
        super().__init__(classes)
        self.num_classes = len(classes)
        self.cm = []
        self.eye = np.eye(self.num_classes, self.num_classes)

    def compute_cm(self, true_labels, predicted_labels):
        return np.einsum('nd,ne->de', self.eye[true_labels.ravel()], self.eye[predicted_labels.ravel()])

    def compute_confusion_matrix(self, true_labels, predicted_labels):
        if np.sum(true_labels) == 0 and np.sum(predicted_labels) == 0:
            return np.zeros(shape=(self.num_classes, self.num_classes))
        try:
            cm = self.compute_cm(true_labels, predicted_labels)
        except MemoryError:
            s = 8
            while 1:
                error = False
                cm = np.zeros_like(self.eye)
                for tl, pl in zip(np.array_split(true_labels.ravel(), s), np.array_split(predicted_labels.ravel(), s)):
                    try:
                        cm_ = self.compute_cm(tl, pl)
                    except MemoryError:
                        s *= 2
                        error = True
                        break
                    cm += cm_
                if not error:
                    break
        return cm

    def _evaluate(self, segmentation, prediction, prob_maps, mask, extra_maps):
        self.cm.append(self.compute_confusion_matrix(segmentation, prediction))
