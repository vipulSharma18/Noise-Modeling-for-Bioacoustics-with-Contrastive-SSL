"""
Test that methods of shuffle and learn work and produce expected output shapes.
"""

import os
import torch
import stable_ssl
from ssl_bioacoustics.custom_trainers import ShuffleAndLearnTrainer


def test_shuffle_and_learn_():
    """
    Test using the following module configs:

    Configs:
    -------
    module:
        backbone:
            _target_: stable_ssl.modules.load_backbone
            name: alexnet  # paper had alexnet but it's too old to be acceptable
            num_classes: null
            weights: False
        projector:
            _target_: stable_ssl.modules.MLP
            sizes: [4096, 512, 128]  # 3 views in parallel fed to projector, so dim remains the same.
            activation: ReLU
            batch_norm: False
        projector_classifier:
            _target_: torch.nn.Linear
            in_features: 384  # 128*3 (sizes of proj) 3 views concatenated to pass to classifier.
            out_features: 2
        backbone_classifier:
            _target_: torch.nn.Linear
            in_features: 12288  # 4096*3 for alexnet. views in concatenated to pass to backbone.
            out_features: 2
        """
    dummy_shuffle_and_learn = object.__new__(ShuffleAndLearnTrainer)

    shapes = (2, 3, 4, 3, 70, 112)
    t = torch.ones(1, shapes[-3], shapes[-2], shapes[-1])
    if os.getenv("SLURM_JOB_ID") is not None:
        backbone = stable_ssl.modules.load_backbone(
            name="alexnet",
            num_classes=None,
            weights=None,
            )
    else:
        backbone = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(shapes[-3]*shapes[-2]*shapes[-1], 4096),
            )
    backbone_output_shape = backbone(t).shape[-1]

    num_classes = 2
    projector = stable_ssl.modules.MLP(
        sizes=[backbone_output_shape, 512, 128],
        activation="ReLU",
        batch_norm=False
        )
    projector_classifier = torch.nn.Linear(
        in_features=128*3,
        out_features=num_classes
        )
    backbone_classifier = torch.nn.Linear(
        in_features=backbone_output_shape*3,
        out_features=num_classes
        )

    dummy_shuffle_and_learn.module = {"backbone": backbone,
                                       "projector": projector,
                                      "backbone_classifier": backbone_classifier,
                                      "projector_classifier": projector_classifier,
                                      }

    # Test format_segments_labels
    dummy_shuffle_and_learn.batch = (torch.cat(
        (torch.ones(*shapes),
         torch.zeros(*shapes))
        ),
                                     torch.empty(2*shapes[0]))

    # test processing of segments and labels
    segments, labels = dummy_shuffle_and_learn.format_segments_labels()
    assert segments.shape == (shapes[0]*2*shapes[1], shapes[-4] - 1, shapes[-3], shapes[-2], shapes[-1])
    assert labels.shape == (shapes[0]*2*shapes[1],)

    # test proper concatenation of types of chunks in forward
    forward_output = dummy_shuffle_and_learn.forward(segments)
    assert forward_output.shape == (shapes[0]*2*shapes[1], 3*backbone_output_shape)

    # test that the classifiers get the concatenated output
    predict_output = dummy_shuffle_and_learn.predict()
    assert predict_output.shape == (shapes[0]*2*shapes[1], num_classes)

    # check class_weights are good
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / class_counts
    print(f"class_weights: {class_weights}")

    # test the projections and embeddings are getting concatenated properly and that the class_weights don't break loss computation
    loss = dummy_shuffle_and_learn.compute_loss()

    assert loss is not None
    print(f"loss: {loss}")
