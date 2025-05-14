"""
Some trainers like shuffle and learn require that the different "views" be
concatenated along the 0th axis instead of contrasted with each other by
having them in a list.

One could do this concatenation in the trainer's forward method, but it's
better to keep the trainer agnostic of the number of views and let the
sampler of the transforms handle it.
"""

import logging
import torch


class ConcatViewsSampler:
    """
    Apply a list of transforms to an input and
    return all outputs concatenated together.
    """

    def __init__(self, transforms: list):
        logging.info(
            f"ConcatViewSampler initialized with {len(transforms)} views"
            )
        self.transforms = transforms

    def __call__(self, x):
        views = []
        for t in self.transforms:
            views.append(t(x))
        if len(self.transforms) == 1:
            return views[0]
        views = torch.stack(views)
        return views
