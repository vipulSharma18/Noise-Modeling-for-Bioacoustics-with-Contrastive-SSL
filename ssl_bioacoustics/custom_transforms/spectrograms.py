"""
Central class to create:
* Mel, STFT, CQT and CWT spectrograms.
* MFCC cepstrogram.
Easier for maintenance of code.
"""

import math
import sklearn
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pywt
import librosa
import torchaudio


class Spectrogram(torch.nn.Module):
    """
    Transform to create mel, stft or cqt spectrograms.

    Parameters:
    -----------
    sampling_rate: int, optional
        The sampling rate of the audio input to the spectrogram.
    representation: str
        Return a spectrogram or cepstrogram.
        We can select a power, amplitude, or complex spectrogram.
        Also supports mfcc and CWT.
        Supports: ['complex_mel', 'power_mel', 'amplitude_mel',
        'complex_stft', 'power_stft', 'amplitude_stft',
        'complex_cqt', 'power_cqt', 'amplitude_cqt',
        'complex_cwt', 'power_cwt', 'amplitude_cwt',
        'mfcc']
    convert_to_db: bool, optional
        If True, convert the spectrogram to decibels wrt np.max.
    representation_mode: str, optional
        (Default) If 'raw' return the raw spectrogram
        converted to a 3D np array with shape (1, n_freq, n_time).
        If 'grayscale', return a single channel PIL image normalized to [0-255].
        If 'rgb', return a 3 channel PIL image by first displaying it
        using librosa.display.specshow.
    """
    def __init__(
        self,
        sampling_rate=22000,
        representation="power_mel",
        convert_to_db=False,
        representation_mode="raw",
        compute_on_gpu_if_available=False,
        **kwargs,
    ):
        super(Spectrogram, self).__init__()
        self.sampling_rate = sampling_rate
        self.representation = representation

        representation_options = ['complex_mel', 'power_mel', 'amplitude_mel',
                                  'complex_stft', 'power_stft', 'amplitude_stft',
                                  'complex_cqt', 'power_cqt', 'amplitude_cqt',
                                  'complex_cwt', 'power_cwt', 'amplitude_cwt',
                                  'mfcc'
                                  ]
        if self.representation not in representation_options:
            raise ValueError(
                f"Unsupported representation {self.representation}. \
                    Supported options are {representation_options}."
                )

        self.representation_mode = representation_mode

        if "complex" in self.representation and self.representation_mode != "raw":
            raise ValueError(
                f"Complex representations can not be represented as grayscale or rgb images. \
                    Use the amplitude or power representations instead."
                )
        self.power = None
        if "power" in self.representation:
            self.power = 2.0
        elif "amplitude" in self.representation:
            self.power = 1.0
        elif "complex" in self.representation:
            self.power = None
        self.convert_to_db = convert_to_db

        self.kwargs = kwargs
        self.compute_on_gpu_if_available = compute_on_gpu_if_available
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.compute_on_gpu_if_available else "cpu")

    def add_noise_to_spectrogram(self, spectrogram, noise_level=0.1):
        """
        Add noise directly to a spectrogram.
        
        Parameters:
        -----------
        spectrogram: numpy.ndarray or torch.Tensor
            The input spectrogram
        noise_level: float
            The level of noise to add (0-1)
            
        Returns:
        --------
        numpy.ndarray or torch.Tensor
            The spectrogram with added noise
        """
        if isinstance(spectrogram, torch.Tensor):
            device = spectrogram.device
            spectrogram = spectrogram.cpu().numpy()
            
        # Generate noise with same shape as spectrogram
        noise = np.random.normal(0, noise_level, spectrogram.shape)
        
        # Add noise to spectrogram
        noisy_spectrogram = spectrogram + noise
        
        # Convert back to tensor if input was tensor
        if isinstance(spectrogram, torch.Tensor):
            noisy_spectrogram = torch.from_numpy(noisy_spectrogram).to(device)
            
        return noisy_spectrogram

    def forward(self, x):
        """
        Pick appropriate representation for an input
        or return it as a waveform.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if "mel" in self.representation:
            n_fft = self.kwargs.get("n_fft", 2048)
            win_length = self.kwargs.get("win_length", n_fft)
            hop_length = self.kwargs.get("hop_length", win_length // 4)
            n_mels = self.kwargs.get("n_mels", 256)
            fmin = self.kwargs.get("fmin", 0)
            fmax = self.kwargs.get("fmax", self.sampling_rate / 2.0)

            x = torch.from_numpy(x).to(self.device)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sampling_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=fmin,
                f_max=fmax,
                n_mels=n_mels,
                power=self.power,
                norm="slaney",
                mel_scale="htk").to(self.device)
            x = mel_transform(x).cpu().numpy()
            # stft_matrix = librosa.stft(y=x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            # mel_basis = librosa.filters.mel(sr=self.sampling_rate,
            #                                 n_fft=n_fft, n_mels=n_mels,
            #                                 fmin=fmin, fmax=fmax)
            # x = np.dot(mel_basis, stft_matrix)  # (..., n_mels, n_time)
        elif "stft" in self.representation:
            n_fft = self.kwargs.get("n_fft", 2048)
            win_length = self.kwargs.get("win_length", n_fft)
            hop_length = self.kwargs.get("hop_length", win_length // 4)
            x = librosa.stft(y=x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        elif "mfcc" in self.representation:
            n_mfcc = self.kwargs.get("n_mfcc", 40)
            # melspectrogram params
            n_fft = self.kwargs.get("n_fft", 2048)
            win_length = self.kwargs.get("win_length", n_fft)
            # mel filterbank default params
            n_mels = self.kwargs.get("n_mels", 256)
            fmin = self.kwargs.get("fmin", 0)
            fmax = self.kwargs.get("fmax", self.sampling_rate / 2.0)
            x = librosa.feature.mfcc(y=x, sr=self.sampling_rate, n_mfcc=n_mfcc,
                                     n_mels=n_mels, fmax=fmax, fmin=fmin,
                                     n_fft=n_fft, win_length=win_length,
                                     )
            x = sklearn.preprocessing.scale(x, axis=1)
        elif "cwt" in self.representation:
            # B is bandwidth, C is center frequency. w_0 = 2 * pi * C, usually 6.
            # instead of using sampling period of 1/samplig rate, we use a large enough scale range from 1 to sampling rate/2.
            wavelet = self.kwargs.get("wavelet", "cmor2-0.955")  # cmorB-C
            n_scales = self.kwargs.get("n_scales", 84)  # J, or total rows/scales in output.
            widths = np.geomspace(1, self.sampling_rate / 2, n_scales)
            x, _ = pywt.cwt(x, scales=widths, wavelet=wavelet)
        elif "cqt" in self.representation:
            baseline_note = self.kwargs.get("baseline_note", "C1")
            fmin = librosa.note_to_hz(baseline_note)
            # librosa.interval_frequencies(n_bins, fmin, bins_per_octave) will give the max sampled freq
            # frequencies get doubled every octave.
            # freq get 2*(i/num_of_bins_per_octave) every bin within an octave.
            # ith bin within an octave: 2^(i/num_of_bins_per_octave) * f_octave
            # f_octave for jth octave: 2^(j) * f_min
            # num_of_octaves = ceil(num_bins/bins_per_octave)
            # f_min = 32.70 Hz for C1.
            # f_max = sampling_rate/2.0
            # final f: 2^(j) * 2^(i/bins_per_octave) * f_min
            # 2^(256/bins_per_octave) * f_min <= sampling_rate/2
            freq_bins = self.kwargs.get("freq_bins", 256)
            bins_per_octave = math.ceil(
                freq_bins/(math.log(self.sampling_rate/(2 * fmin), 2))
                )
            x = librosa.cqt(
                    x,
                    sr=self.sampling_rate,
                    fmin=fmin,
                    n_bins=freq_bins,
                    bins_per_octave=bins_per_octave,
                    )
        else:
            raise ValueError(
                f"Unsupported representation {self.representation}."
                )
        x = self.format_representation(x)

        return x

    def format_representation(self, x):
        if (self.representation == "mfcc") or ("mel" in self.representation):
            pass
        elif self.power is not None:
            x = np.abs(x) ** self.power

        if self.representation == "mfcc":
            pass
        elif self.convert_to_db:
            ref = torch.tensor(np.max(x))
            x = torch.from_numpy(x)
            if self.power == 2.0:
                # x = librosa.power_to_db(x, ref=np.max)  # this computes wrt to peak power (np.max) in the signal.
                x = torchaudio.functional.amplitude_to_DB(
                    x, multiplier=10., amin=1e-10, top_db=80.0, db_multiplier=torch.log10(max(ref, torch.tensor(1e-10)))
                    )
            elif self.power == 1.0:
                # x = librosa.amplitude_to_db(x, ref=np.max)
                x = torchaudio.functional.amplitude_to_DB(
                    x, multiplier=20., amin=1e-5, top_db=80.0, db_multiplier=torch.log10(max(ref, torch.tensor(1e-5)))
                    )
            x = x.cpu().numpy()

        # convert the representation to either an image or a 3D tensor
        if self.representation_mode == "raw":
            if len(x.shape) == 2:
                # just add 1 channel to make it a 3D tensor
                # and support CNNs + transforms.
                x = np.flipud(x)  # put the lowest frequency at the bottom like librosa.display.specshow
                x = np.expand_dims(x, axis=0)
        elif self.representation_mode == "rgb":
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            # x-axis: time, y-axis: frequency
            _ = librosa.display.specshow(
                x,
                sr=self.sampling_rate,
                ax=ax,
                )
            fig.canvas.draw()
            x = Image.frombytes(
                'RGBA',
                fig.canvas.get_width_height(),
                fig.canvas.buffer_rgba(),
                )
            x = x.convert("RGB")
            plt.close(fig)
        elif self.representation_mode == "grayscale":
            # print(x.shape, type(x))
            x = np.flipud(x)  # put the lowest frequency at the bottom like librosa.display.specshow. otherwise RGB will be upside down of grayscale.
            x = Image.fromarray(
                (255 * (x - np.min(x)) / (np.max(x) - np.min(x))).astype(
                    np.uint8
                    )
            )
        else:
            raise ValueError(
                f"Unsupported representation mode {self.representation_mode}."
                )

        return x
