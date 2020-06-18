from abc import ABC
import numpy as np
import torch
import torchvision


def to_np_cpu(tensor):
    return tensor.to('cpu').detach().numpy()


def report_scalar(name, scalar):
    return '{0}: {1:.4f}\t'.format(name, scalar).ljust(20)


def report_mean_and_std(name, array):
    return '{}: {:.4f} ± {:.4f}\t'.format(name, np.mean(array), np.std(array))


class Metric(ABC):
    def __init__(self, initial_value):
        self.running_value = initial_value
        self.value = None

    def increment(self, model_state):
        raise NotImplementedError

    def save_and_reset(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def log_to_tensorboard(self, epoch, writer, tag):
        raise NotImplementedError


class Loss(Metric):
    def __init__(self, device):
        Metric.__init__(self, torch.zeros(1, device=device))

    def increment(self, model_state):
        self.running_value += model_state['loss']

    def save_and_reset(self):
        self.value = float(self.running_value.to('cpu').detach())
        self.running_value[:] = 0

    def report(self):
        return report_scalar('loss', self.value) + '\n'

    def log_to_tensorboard(self, epoch, writer, tag):
        writer.add_scalar(tag, self.value, global_step=epoch)


class MultiChannelTensorDistribution(Metric):
    # assumes shape (batch_size, num_maps, ...)
    def __init__(self, name, num_maps: int):
        self.name = name
        Metric.__init__(self, [[0, 0, 0] for _ in range(num_maps)])
        self.num_maps = num_maps
        self.value = [None for _ in range(num_maps)]

    def increment(self, model_state):
        tensor = model_state[self.name].detach()
        for i in range(self.num_maps):
            t = tensor[:, i]
            self.running_value[i][0] += t.numel()
            self.running_value[i][1] += torch.sum(t)
            self.running_value[i][2] += torch.sum(t * t)

    def save_and_reset(self):
        for i in range(self.num_maps):
            n = self.running_value[i][0]
            sum_ = self.running_value[i][1]
            sum_square = self.running_value[i][2]
            var = (sum_square - ((sum_ ** 2) / n)) / (n - 1)
            mean = sum_ / n
            self.value[i] = [mean.cpu().numpy(), torch.sqrt(var).cpu().numpy()]
        self.running_value = [[0, 0, 0] for _ in range(self.num_maps)]

    def report(self):
        messages = []
        for i, value in enumerate(self.value):
            messages.append(f'{self.name:s}_{i:d}: {value[0]:.4f} ± {value[1]:.4f}')
        l = max([len(m) for m in messages])
        message = ''
        for m in messages:
            message += m.ljust(l) + '\t'
        message += '\n'
        return message

    def log_to_tensorboard(self, epoch, writer, tag):
        for i, value in enumerate(self.value):
            writer.add_scalar(f'{tag:s}_{i:d}/mean', value[0], global_step=epoch)
            writer.add_scalar(f'{tag:s}_{i:d}/stddev', value[1], global_step=epoch)


class TrackTensor(Metric):

    def __init__(self, name):
        self.name = name
        Metric.__init__(self, [])

    def increment(self, model_state):
        self.value += list(model_state[self.name])

    def save_and_reset(self):
        self.value = to_np_cpu(torch.squeeze(torch.stack(tuple(self.running_value))))
        self.running_value = []

    def report(self):
        return report_mean_and_std(self.name, self.value)

    def log_to_tensorboard(self, epoch, writer, tag):
        pass


class RunningConfusionMatrix(Metric):

    def __init__(self, num_classes, device):
        super().__init__(torch.zeros((num_classes, num_classes), device=device))
        self.eye = torch.eye(num_classes, num_classes, device=device)

    def compute_confusion_matrix(self, labels, preds):
        return torch.einsum('nd,ne->de', self.eye[labels.flatten()], self.eye[preds.flatten()])

    def increment(self, model_state):
        self.running_value += self.compute_confusion_matrix(model_state['target'], model_state['pred'])

    def save_and_reset(self):
        self.value = to_np_cpu(self.running_value)
        self.running_value[:] = 0

    def report(self):
        return ''

    def log_to_tensorboard(self, epoch, writer, tag):
        pass


def calc_accuracy(cm):
    return np.repeat(np.sum(np.diag(cm)) / np.sum(cm), cm.shape[0])


def calc_precision(cm):
    return np.diag(cm) / np.sum(cm, axis=0)


def calc_recall(cm):
    return np.diag(cm) / np.sum(cm, axis=1)


def calc_f1_score(cm):
    return np.diag(cm) / np.sum(cm, axis=1)


class ClassificationMetrics(RunningConfusionMatrix):
    metric_fns = {'accuracy': calc_accuracy,
                  'precision': calc_precision,
                  'recall': calc_recall,
                  'f1_score': calc_f1_score}

    def __init__(self, device, class_names):

        self.class_names = class_names

        RunningConfusionMatrix.__init__(self, len(class_names), device)
        self.metrics = dict.fromkeys(self.metric_fns.keys())

    def save_and_reset(self):
        super().save_and_reset()
        cm = self.value
        for metric in self.metrics:
            self.metrics[metric] = self.metric_fns[metric](cm)

    def log_to_tensorboard(self, epoch, writer, tag):
        for i, class_name in enumerate(self.class_names):
            for metric, value in self.metrics.items():
                writer.add_scalar(class_name + '/' + str(metric), value[i], global_step=epoch)

    def report(self):
        message = ''
        for i, class_name in enumerate(self.class_names):
            message += class_name.upper().ljust(20) + ':\t'
            for metric, value in self.metrics.items():
                message += report_scalar(metric, value[i])
            message += '\n'
        return message


class SegmentationMetrics(ClassificationMetrics):
    def __init__(self, device, class_names):
        class_names[0] = 'Foreground'
        super().__init__(device, class_names)

    @staticmethod
    def merge_cm_classes(cm):
        new_cm = np.zeros(shape=(2, 2))
        fi = list(range(1, cm.shape[0]))
        new_cm[1, 1] = np.sum(cm[fi, fi])
        new_cm[0, 0] = cm[0, 0]
        new_cm[1, 0] = np.sum(cm[fi, 0])
        new_cm[0, 1] = np.sum(cm[0, fi])
        return new_cm

    def save_and_reset(self):
        super().save_and_reset()
        cm = self.value
        f_cm = self.merge_cm_classes(cm)
        for metric in self.metrics:
            self.metrics[metric][0] = self.metric_fns[metric](f_cm)[1]


class SegmentationImageThumbs(Metric):
    def __init__(self, num_images, channel_idx=0, opacity=.5, keep_every=20):
        super().__init__(None)
        self.num_images = num_images
        self.channel_idx = channel_idx
        self.opacity = opacity
        self.keep_every = keep_every  # save memory in the logs
        self.color_map = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)))

    def increment(self, model_state):
        if self.running_value is None and model_state['epoch'] % self.keep_every == 0:
            batch_size = len(model_state['image'])
            assert self.num_images <= batch_size
            self.running_value = {'image': model_state['image'][:self.num_images, self.channel_idx],
                                  'target': model_state['target'][:self.num_images],
                                  'pred': model_state['pred'][:self.num_images]}

    @staticmethod
    def crop_center(x, size):
        crop = (slice(0, x.shape[0], 1), )
        crop += tuple(slice(c // 2 - s // 2, c // 2 + s // 2 + s % 2, 1) for c, s in zip(x.shape[1:], size))
        return x[crop]

    def get_overlay(self, binary_images):
        overlay = self.color_map[binary_images]
        overlay = overlay.transpose((0, -1) + tuple(range(1, len(overlay.shape) - 1)))
        return overlay

    def mix_image_and_overlay(self, image, overlay):
        new_image = np.copy(image)
        ind = np.broadcast_to(np.sum(overlay, axis=1, keepdims=True) > 0, image.shape)
        new_image[ind] = self.opacity * overlay[ind] + (1 - self.opacity) * image[ind]
        return new_image

    def save_and_reset(self):
        if self.running_value is None:
            return
        images = self.running_value['image'].cpu().numpy()
        target = self.running_value['target'].cpu().numpy().astype(np.uint8)
        prediction = self.running_value['pred'].cpu().numpy().astype(np.uint8)
        images = self.crop_center(images, target.shape[1:])
        dim = len(images.shape) - 1
        if dim not in [2, 3]:
            raise ValueError(f'Unsupported number of dimensions {dim:d} for showing images.')
        if dim == 3:
            slice_ = np.random.randint(0, images.shape[1])
            images = images[:, slice_]
            target = target[:, slice_]
            prediction = prediction[:, slice_]
        mins = np.min(images, axis=(1, 2), keepdims=True)
        maxs = np.max(images, axis=(1, 2), keepdims=True)
        images = (images - mins) / (maxs - mins)
        rgb_images = np.stack((images,) * 3, axis=1)
        images_with_target = self.mix_image_and_overlay(rgb_images, self.get_overlay(target))
        images_with_prediction = self.mix_image_and_overlay(rgb_images, self.get_overlay(prediction))
        full_image = torch.Tensor(np.concatenate((rgb_images, images_with_target, images_with_prediction)))
        self.value = torchvision.utils.make_grid(full_image, nrow=self.num_images)
        self.running_value = None

    def report(self):
        return ''

    def log_to_tensorboard(self, epoch, writer, tag):
        if epoch % self.keep_every == 0 and self.value is not None:
            writer.add_image(tag, self.value, epoch)
