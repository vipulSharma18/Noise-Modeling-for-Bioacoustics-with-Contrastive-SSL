"""
Common visualization functions for audio data.
The functions are adapted from torchaudio's tutorial:
https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_waveform(waveform, sr, title="Waveform", axis=None):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if len(waveform.shape) == 1:
        # single channel audio loaded by librosa
        waveform = np.expand_dims(waveform, axis=0)

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if axis is None:
        _, axis = plt.subplots(num_channels, 1)
    axis.plot(time_axis, waveform[0], linewidth=1)
    axis.grid(True)
    axis.set_xlim([0, time_axis[-1]])
    axis.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="Freq Bin", axis=None, pil_image=False):
    if not pil_image and isinstance(specgram, torch.Tensor):
        specgram = specgram.numpy()
    # either multi-channel or dummy channel dimension
    if not pil_image and len(specgram.shape) > 2:
        specgram = specgram[0]  # only plot the first channel
    if axis is None:
        _, axis = plt.subplots(1, 1)
    if title is not None:
        axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.imshow(
        specgram,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
        )


def plot_fbank(fbank, title=None, axis=None):
    if isinstance(fbank, torch.Tensor):
        fbank = fbank.numpy()
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    axis.set_title(title or "Filter bank")
    axis.imshow(fbank, aspect="auto")
    axis.set_ylabel("frequency bin")
    axis.set_xlabel("mel bin")


def plot_pitch(waveform, sr, pitch, title, axis=None):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if len(waveform.shape) == 1:  # single channel audio loaded by librosa
        waveform = np.expand_dims(waveform, axis=0)
    if isinstance(pitch, torch.Tensor):
        pitch = pitch.numpy()
    if len(pitch.shape) == 1:
        pitch = np.expand_dims(pitch, axis=0)
    if axis is None:
        figure, axis = plt.subplots(1, 1)
    axis.set_title(title or "Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)
