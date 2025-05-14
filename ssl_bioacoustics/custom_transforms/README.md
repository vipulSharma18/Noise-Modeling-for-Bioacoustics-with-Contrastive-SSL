## Literature on representation and performance:
* Huzaifah, 2017 does a comparison of different ways to get a spectrogram representation and the impact on sound classification performance.
* MFCC features were used as baseline.
* STFT w/t linear and Mel scales spectrogram, CQT spectrogram, CWT (continuous wavelet transform) scalogram: Mel-scaled STFT was the best representation.
* For corruption directly on spectrogram, the inversion to audio waveform will be lossy as the spectrogram doesn't have phase information which is estimated using a vocoder or Griffin Lim algo, etc. Func: librosa.feature.inverse.mel_to_audio or torchaudio.transforms.GriffinLim .
* CWT generation methodology is same as Huzaifah, 2017, who in turn follows:
    * O. Rioul and M. Vetterli, “Wavelets and signal processing,” IEEE signal processing magazine, vol. 8, no. 4, pp. 14–38, 1991.
    * M. C. Orr, D. S. Pham, B. Lithgow, and R. Mahony, “Speech perception based algorithm for the separation of overlapping speech signal,” in Intelligent Information Systems Conference, The Seventh Australian and New Zealand 2001. IEEE, 2001, pp. 341–344.
    * M. Cowling and R. Sitte, “Comparison of techniques for environmental sound recognition,” Pattern recognition letters, vol. 24, no. 15, pp. 28952907, 2003.

## Natural Corruptions:
### UrbanSound8k:
* Has sounds from urban acoustic environment. Creates an imagenet like taxonomy for audio.
* We can also use the full Urban Sound instead of the 8k samples.
* Superclasses: Human, Nature, Mechanical, Music.
* Human: Voice and movement.
* Nature: Elements (wind), animals (bark), and plants/vegetaion (leaves rustling).
* Mechanical: Construction, ventilation, non-motorized transport, social/signals (bells, alarms), motorized transport, music.
* classID:
    A numeric identifier of the sound class:
    0 = air_conditioner
    1 = car_horn
    2 = children_playing
    3 = dog_bark
    4 = drilling
    5 = engine_idling
    6 = gun_shot
    7 = jackhammer
    8 = siren
    9 = street_music

### VROOM:
* Motor, vehicles and power tools related sounds which the dataset owners have claimed are missing in UrbanSound8k. However, there are motorized vehicles sounds in the UrbanSound8k taxonomoy diagram.

## Artificial Corruptions (starting suggestions by chat-gpt and then torchaudio and librosa surveyed for possible transforms):
### Time-Domain Augmentations:
* Time Stretching/Compression: Changing the speed (tempo) of the audio without altering the pitch. This helps the model become invariant to variations in tempo.
    * torchaudio.transforms.Speed(orig_freq, factor>1.0 compress, <1.0 stretches)
    * torchaudio.transforms.Speed().forward(waveform, lengths of valid waveform)
    * can vary the factor at runtime using torchaudio.transforms.SpeedPerturbation(orig_freq, factors: list from which to uniformly sample)
* Adding Noise: Superimposing random noise (e.g., Gaussian noise) on the signal to mimic real-world variations and improve robustness.
    * torchaudio.transforms.AddNoise().forward(waveform, noise, snr, lengths optional - valid length of signals waveform and noise)
    * add gaussian here and for natural use urbansound8k and vroom.
    * -snr means more noise is added. around -5 is reasonable...

### Frequency-Domain Augmentations:
* Pitch Shifting: Adjusting the pitch up or down without affecting the tempo. This can be done by resampling or using more sophisticated techniques like phase vocoders.
    * torchaudio.transforms.PitchShift(sample_rate, n_steps).forward(waveform) where n_steps is the number of semitones to shift the pitch.
    * by default n_bins_per_octave, or the number of steps per octave is 12.
* SpecAugment: A widely used method that applies augmentation directly to the spectrogram by masking blocks along the time and/or frequency axes. This includes:
    * Time Masking: Randomly masking consecutive time steps.
        * torchaudio.transforms.TimeMasking(max possible length of mask but mask can be smaller than this, iid_masks: default False, iid for diff examples in the batch., p: proportion [0,1] of the time steps that can be masked, default=1)
        * works on a spectrogram (Not sure if necessarily complex)
        * mask of size uniform(0, mask_param) where mask_param is mask_param when p=1, else it is min(mask_param, p*total length of spectrogram)
    * Frequency Masking: Randomly masking consecutive frequency bins.
        * torchaudio.transforms.FrequencyMasking(mask_param, iid_masks)
        * no p here, the mask is always of size uniform(0, mask_param)
    * Time Warping: Distorting the time axis of the spectrogram (less common but sometimes used).
        * torchaudio.transforms.TimeStretch().forward(waveform, factor) -> will change length 
        * works on a complex spectrogram, i.e., output of torchaudio.transforms.Spectrogram(power=None) or librosa.STFT()

### Dynamic Range and Amplitude Adjustments:
* Volume Control/Amplitude Scaling: Randomly adjusting the overall gain or amplitude of the signal.
    * torchaudio.transforms.Vol(gain: float, gain_type: str one of ['amplitude', 'power', 'db'])

## Dataset specific representation parameters:
### Cornell Bird Challenge 2020:
* 


## References:
1. Transforms available in torchaudio: https://pytorch.org/audio/main/transforms.html
2. Transforms in librosa: https://librosa.org/doc/main/effects.html# 
3. Huzaifah, M. (2017) ‘Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks’. arXiv. Available at: https://doi.org/10.48550/arXiv.1706.07156.
