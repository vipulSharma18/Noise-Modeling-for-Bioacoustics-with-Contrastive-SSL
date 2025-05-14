"""
Shuffle and Learn SSL trainer.
Code is developed by referring to:
[1] BizhuWu, BizhuWu/ShuffleAndLearn_PyTorch. (Apr. 22, 2024). Python. Available: https://github.com/BizhuWu/ShuffleAndLearn_PyTorch
[2] I. Misra, C. L. Zitnick, and M. Hebert, “Shuffle and Learn: Unsupervised Learning using Temporal Order Verification,” Jul. 26, 2016, arXiv: arXiv:1603.08561. doi: 10.48550/arXiv.1603.08561.
[3] P. Ryan, SingingData/Birdsong-Self-Supervised-Learning. (Mar. 25, 2022). Jupyter Notebook. Available: https://github.com/SingingData/Birdsong-Self-Supervised-Learning
"""

import torch
import torch.nn.functional as F

from stable_ssl.base import BaseTrainer
from stable_ssl.utils import log_and_raise


class ShuffleAndLearnTrainer(BaseTrainer):
    """
    Base class for training a Shuffle and Learn SSL model.
    It uses temporal order verification to learn representations.

    Dataset will have 1 audio sample and possibly a label.
    The sample might or might not be randomly selected as per shuffle.
    We'll then use a transform to chunk and shuffle the audio sample to create
    temporally correct and shuffled segments along with a 1/0 label.
    """

    required_modules = {
        "backbone": torch.nn.Module,
        "projector": torch.nn.Module,
        "backbone_classifier": torch.nn.Module,
        "projector_classifier": torch.nn.Module,
    }

    def format_segments_labels(self):
        """
        Take the batch and convert it into the right (x,y) tuple.
        batch = ((batch_dim), (num_of_shuffles/views), (len of chunks + 1 for label), (size of spectrogram), dummy_label)
        Example:
        --------
        batch[0]: torch.Size([240, 3, 4, 1, 70, 112])
        """
        if (
            len(self.batch) == 2  # x, y where y could be an empty tensor
            # all the different views (different temporal orders) are concatenated
            and torch.is_tensor(self.batch[0][0])
            and (len(torch.unique(self.batch[0][0][0][-1])) == 1)
        ):
            # we assume the second element, batch[1], will be the original label which we can ignore
            seg_labels = self.batch[0]  # batch_sz, n_types_shuffles, n_chunks+1, size of spectrogram (n_freq, n_time)
            segments = seg_labels[:, :, :-1]
            segments_shape = segments.shape
            segments = segments.view(
                segments_shape[0] * segments_shape[1],
                *segments_shape[2:],
                )
            labels = seg_labels[:, :, -1].squeeze().long()
            labels = labels.view(labels.size(0), labels.size(1), -1)
            # more efficient than this as all are the same items: torch.unique(labels, dim=-1).squeeze()
            labels = labels[..., 0].squeeze()
            labels_shape = labels.shape
            labels = labels.view(
                labels_shape[0] * labels_shape[1],
                *labels_shape[2:],
                )
            # to shuffle the positive and negative samples within the batch (should not make difference ideally)
            indices = torch.randperm(labels.size(0))
            segments, labels = segments[indices], labels[indices]
        else:
            msg = "Shuffle and learn got unexpected input! "
            msg = msg + f"Got: len batch {len(self.batch)}, "
            msg = msg + f"batch[0] {type(self.batch[0])}, "
            msg = msg + f"batch[1] {type(self.batch[1])}. "
            msg = msg + f"torch.is_tensor(self.batch[0]): {torch.is_tensor(self.batch[0])}, "
            msg = msg + f"len(torch.unique(self.batch[0][0][-1])): {len(torch.unique(self.batch[0][0][-1]))}. "
            msg = msg + f"batch[0]: {self.batch[0].shape}, {self.batch[0][0].shape}, {self.batch[0][0][-1].shape}"
            log_and_raise(ValueError, msg)
        return segments, labels

    def forward(self, segments):
        """
        Forward pass. By default, it simply calls the 'backbone' module.
        This is used as part of predict during the eval mode. Not used during training.
        """
        # parallel pass on all the segments of a sequence
        total_chunks = segments.size(1)
        outputs = []
        for i in range(total_chunks):
            outputs.append(
                self.module["backbone"](segments[:, i])
            )
        # number of chunks, batch_sz, model_output_size
        # batch, total_chunks * model_output_size
        output = torch.cat(outputs, dim=1)
        return output

    def predict(self):
        """
        Call the backbone classifier on the forward pass of current batch.
        This is used for evaluation while compute_loss is used for training.
        """
        segments, labels = self.format_segments_labels()
        backbone_output = self.forward(segments)
        return self.module["backbone_classifier"](backbone_output)

    def compute_loss(self):
        """
        Compute final loss as sum of SSL loss and classifier losses.
        Custom self.loss isn't required for this as it's by definition
        a temporal order verification task w/t positive and negative labels.
        """
        segments, labels = self.format_segments_labels()
        # parallel pass on all the segments of a sequence
        total_chunks = segments.size(1)
        individual_embeddings = []
        for i in range(total_chunks):
            individual_embeddings.append(
                self.module["backbone"](segments[:, i])
            )  # number of chunks, batch_sz, model_output_size
        embeddings = torch.cat(individual_embeddings, dim=1)
        self.latest_forward = embeddings
        individual_projections = [
            self.module["projector"](individual_embeddings[i])
            for i in range(total_chunks)
            ]
        projections = torch.cat(individual_projections, dim=1)

        classifier_losses = self.compute_loss_classifiers(
            embeddings, projections, labels
        )

        return {**classifier_losses}

    def compute_loss_classifiers(self, embeddings, projections, labels):
        """Compute the classifier loss for both backbone and projector."""
        loss_backbone_classifier = 0
        loss_projector_classifier = 0

        # Inputs are detached to avoid backprop through backbone and projector.
        if labels is not None:
            class_counts = torch.bincount(labels)  # Counts the occurrences of each class (0 and 1)
            print("class_counts:", class_counts, "uniq labels:", torch.unique(labels))
            total_samples = len(labels)
            class_weights = total_samples / class_counts  # Inverse proportional weights for binary classes
            # reproducing the paper's results and they didn't do normalization. If we normalize, we may need to adjust hyperparams.
            # class_weights = class_weights / class_weights.sum()  # normalize to sum to 1
            class_weights = class_weights.to(labels.device)
            print("class_weights:", class_weights)

            loss_backbone_classifier += F.cross_entropy(
                self.module["backbone_classifier"](embeddings.detach()),
                labels,
                weight=class_weights,
            )
            loss_projector_classifier += F.cross_entropy(
                self.module["projector_classifier"](projections.detach()),
                labels,
                weight=class_weights,
            )
        else:
            msg = "Shuffle and learn expects temporal order verification \
                labels but none were received."
            log_and_raise(ValueError, msg)

        return {
            "loss_backbone_classifier": loss_backbone_classifier,
            "loss_projector_classifier": loss_projector_classifier,
        }
