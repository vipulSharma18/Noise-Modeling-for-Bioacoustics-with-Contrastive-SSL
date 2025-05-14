"""
Contrastive Predictive Coding trainer.
Reference:
https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py
"""

import torch
import torch.nn.functional as F

from stable_ssl.base import BaseTrainer
from stable_ssl.utils import log_and_raise


class SlidingCPCTrainer(BaseTrainer):
    """
    TLDR: It's like a 3 size shifting window CNN/RWKV/Canon type layer.

    Base class for training a CPC model with sliding window.
    Unlike typical CPC, which has steps of 10-20ms, our windows are much larger.
    
    So I'm doing a sliding window style CPC, which takes a step and contrasts it with the next 2 steps.
    This is done for every step in the sequence.

    Input will be 1 audio (spectrogram representation) sample split into multiple slides.
    """

    def __init__(self, convolution_window: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolution_window = convolution_window

    required_modules = {
        "backbone": torch.nn.Module,
        "projector": torch.nn.Module,
        "backbone_classifier": torch.nn.Module,
    }

    def forward(self, *args, **kwargs):
        """
        Forward pass. By default, it simply calls the 'backbone' module.
        This is used as part of predict during the eval mode. Not used during training.
        """
        # parallel pass on all the segments of a sequence. forward pass CNN_LSTM.
        outputs = self.module["backbone"](*args, **kwargs)
        return outputs

    def predict(self):
        """
        Call the backbone classifier on the forward pass of current batch.
        This is used for evaluation while compute_loss is used for training.
        """
        backbone_output = self.forward(self.batch[0])
        backbone_output = backbone_output.sum(dim=1)  # aggregate across timesteps
        return self.module["backbone_classifier"](backbone_output)

    def compute_loss(self):
        """
        Compute final loss as sum of SSL loss and classifier losses.
        This function just calculates the CPC loss
        and defers the classifier loss to compute_loss_classifiers.
        """
        backbone_output = self.module["backbone"](self.batch[0])  # B, Timesteps, D
        self.latest_forward = backbone_output

        # CPC loss with anchor as current context, positive as future comparison_steps, negative as other samples in the batch.
        projections_timestep = [self.module["projector"](backbone_output[:, i]) for i in range(backbone_output.shape[1])]
        projections_timestep = torch.stack(list(projections_timestep), dim=1)  # B, Timesteps, D

        loss_ssl = 0
        timesteps = projections_timestep.shape[1]

        for i in range(timesteps-self.convolution_window+1):
            anchors = [projections_timestep[:, i].clone() for _ in range(self.convolution_window-1)]
            comparisons = [projections_timestep[:, i+j].clone() for j in range(1, self.convolution_window)]
            
            for j in range(self.convolution_window-1):
                loss_ssl = loss_ssl + self.loss(anchors[j], comparisons[j])

        classifier_losses = self.compute_loss_classifiers(
            backbone_output, self.batch[1]
        )

        return {"loss_ssl": loss_ssl, **classifier_losses}

    def compute_loss_classifiers(self, backbone_output, labels):
        """Compute the classifier loss for backbone.
        No projector classifier required for us."""
        loss_backbone_classifier = 0

        # Inputs are detached to avoid backprop through backbone and projector.
        if labels is not None:
            x = backbone_output.detach()
            x = x.sum(dim=1)  # aggregate across timesteps
            loss_backbone_classifier = F.cross_entropy(
                self.module["backbone_classifier"](x),
                labels,
            )
        else:
            msg = "Need class label to calculate classification loss."
            log_and_raise(ValueError, msg)

        return {
            "loss_backbone_classifier": loss_backbone_classifier,
        }
