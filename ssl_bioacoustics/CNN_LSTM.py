"""
CNN LSTM models like in Microsoft's Bird-Acoustics-RNN repo,
https://github.com/microsoft/bird-acoustics-rcnn/blob/main/CNN_RNN.ipynb

Code adapted from stable-ssl.
"""

import copy
import logging

import torch
import torch.nn as nn
import torchvision


def load_backbone(
    name,
    num_classes,
    weights=None,
    low_resolution=False,
    return_feature_dim=False,
    **kwargs,
):
    """Load a backbone model.

    If num_classes is provided, the last layer is replaced by a linear layer of
    output size num_classes. Otherwise, the last layer is replaced by an identity layer.

    Parameters
    ----------
    name : str
        Name of the backbone model. Supported models are:
        - Any model from torchvision.models
        - "cnn_lstm"
    num_classes : int
        Number of classes in the dataset.
        If None, the model is loaded without the classifier.
    weights : bool, optional
        Whether to load a weights model, by default False.
    low_resolution : bool, optional
        Whether to adapt the resolution of the model (for CIFAR typically).
        By default False.
    return_feature_dim : bool, optional
        Whether to return the feature dimension of the model.
    **kwargs: dict
        Additional keyword arguments for the model.

    Returns
    -------
    torch.nn.Module
        The neural network model.
    """
    # Load the name.
    if name.lower() == "cnn_lstm":
        model = CNN_LSTM(**kwargs)
    else:
        try:
            model = torchvision.models.__dict__[name](weights=weights, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown model: {name}.")

    # Adapt the last layer, either linear or identity.
    def last_layer(num_classes, in_features):
        if num_classes is not None:
            return nn.Linear(in_features, num_classes)
        else:
            return nn.Identity()

    # For models like ResNet.
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = last_layer(num_classes, in_features)
    # For models like VGG or AlexNet.
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = last_layer(num_classes, in_features)
    # For models like ViT.
    elif hasattr(model, "heads"):
        in_features = model.heads.head.in_features
        model.heads.head = last_layer(num_classes, in_features)
    # For models like Swin Transformer.
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = last_layer(num_classes, in_features)
    else:
        raise ValueError(f"Unknown model structure for : '{name}'.")

    if low_resolution:  # reduce resolution, for instance for CIFAR
        if hasattr(model, "conv1"):
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            logging.warning(f"Cannot adapt resolution for model: {name}.")

    if return_feature_dim:
        return model, in_features
    else:
        return model


class CNN_LSTM(nn.Module):
    """CNN_LSTM model."""

    def __init__(
        self,
        num_classes=100,
        hidden_size=512,
        output_size=512,
        num_layers=2,
        return_hidden_state=False,
        ):
        super().__init__()

        cnnConfig = [32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'] # CNN1

        # cnnConfig = [
        #     64, 64, 'M',\
        #     128, 128, 'M',\
        #     256, 256, 256, 'M',\
        #     512, 512, 512, 'M',\
        #     512, 512, 512, 'M'
        #     ] # CNN3
        self.cnn = self.conv_block(cnnConfig)
        self.n_units = cnnConfig[-2]*2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_hidden_state = return_hidden_state
        self.rnn1 = nn.LSTM(
            input_size=self.n_units,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers
            )
        self.linear1 = nn.Linear(self.hidden_size, output_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(output_size, num_classes)

    @staticmethod
    def conv_block(cnnConfig=None):
        layers = []
        in_channels = 3
        for layer in cnnConfig:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                in_channels = layer
        layers.append(nn.AdaptiveAvgPool2d((2,1)))
        return nn.Sequential(*layers)

    def forward(self, x, hidden_state=None):
        """Forward pass."""
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        x, _ = self.rnn1(r_in, hidden_state)  # batch_first is true
        if self.return_hidden_state:
            # Process each timestep without inplace operations
            processed_states = []
            for i in range(timesteps):
                h = self.linear1(x[:, i])
                h = torch.nn.functional.relu(h)
                h = self.dropout1(h)
                h = self.fc(h)  # num_classes None will make this identity
                processed_states.append(h)
            x = torch.stack(processed_states, dim=1)
            assert x.size(0) == batch_size, f"batch_size mismatch: {x.size(0)} != {batch_size}"
            assert x.size(1) == timesteps, f"timesteps mismatch: {x.size(1)} != {timesteps}"
        else:
            x = x.sum(dim=1)
            x = self.linear1(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout1(x)
            x = self.fc(x)  # num_classes None will make this identity
        return x
