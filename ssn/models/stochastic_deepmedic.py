from .deepmedic import DeepMedic, SCALE_FACTORS, FEATURE_MAPS, FULLY_CONNECTED, DROPOUT
import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
from trainer.distributions import ReshapedDistribution


class StochasticDeepMedic(DeepMedic):
    def __init__(self,
                 input_channels,
                 num_classes,
                 scale_factors=SCALE_FACTORS,
                 feature_maps=FEATURE_MAPS,
                 fully_connected=FULLY_CONNECTED,
                 dropout=DROPOUT,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         feature_maps[-1],
                         scale_factors,
                         feature_maps,
                         fully_connected,
                         dropout)
        conv_fn = nn.Conv3d if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.mean_l = conv_fn(feature_maps[-1], num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(feature_maps[-1], num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(feature_maps[-1], num_classes * rank, kernel_size=(1, ) * self.dim)

    def forward(self, image, **kwargs):
        logits = F.relu(super().forward(image, **kwargs)[0])
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        mask = kwargs['sampling_mask']
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_diag = cov_diag * mask + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = cov_factor.transpose(2, 1).view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict
