# Summary of dataset description:
## Microsoft RCNN Cornell Bird Classification (100 Species):
* https://github.com/microsoft/bird-acoustics-rcnn
* Paper is mostly about how CNN+RNNs are good for birdsong, even better than larger model while having just 1-2 RNN layers.
* They use **mel-spectrogram** as the **input representation**.
* They use the **Cornell Bird Challenge (CBC) 2020** dataset which has **264 birds species' data**.
* Original data has **9-1778 audio samples/species**, with each sample ranging from **5 seconds to 2 mins** in length.
* After thresholding for significant (at least a 100) number of samples per class, we are left with **100 birds species** and **15,032 samples across nearly balanced classes**.
* We use **first 7 seconds** of the samples because xeno-canto requires uploaders to have the bird call within the first few seconds of the audio.
* Pipeline: **Resample to 32kHz. Use 128 mel filter banks, window size of FFT = 2048, hop-length for FFT = 512**. Generate mel-spectrograms (power of freq in db with ref as np.max) using this.
* For temporal models, sliding window mechanism for input spectrogram generation. Window size of 500ms, and hop length of 250ms. Each 7-s audio clip generates 26 sliding and overlapping windows. Each slide/window **input to the model is 128x32 single channel 2-dimensional input (not a RGB spectrogram)**.
* Model performance: **0.57 to 0.67 accuracy** across 100 species, 90% for Red Crossbill bird.

* Inspired by BirdNet, we have used the non-salient audio clips for noise addition augmentation. The noise audio contains ambient sounds.
* Further, we manually select soundscape, rain, and waterfall related recordings from xeno-canto to augment our noise dataset. Each file was split into 7 seconds chunks.


## BirdNet dataset representation details:
* Human and avian vocal and auditory capabilities differ, so we should consider that in representation as mel-spectrogram.
* Birds have **better temporal integration** than humans, and (somehow) that leads to **better** ability to **distinguish the gap** between **two consecutive tones** that differ in **frequency**.
* Due to above, we need **high temporal resolution** (short FFT window length) improves classification performance.
* Spectrograms are considered as **monochrome image**.
* **3-s chunks** -> mel-spectrogram for each chunk.
* FFT window size of 10.7ms (**512 samples at 48kHz**), **overlap of 25%**, i.e., each frame represents a time step of 8ms.
* Bird **vocalization range: 250Hz to 8.3kHz.** Restricted **spectrogram** to be in **150Hz to 15kHz**.
* **64 mel bands** for frequency compression, break frequency at 1750Hz (which gives an approx linear scaling up to 500Hz).
* Augmentations applied with a probability of 0.5 each, and a total of **3 augmentations** out of:
    * Random shifts in frequency & time (vertical and horizontal roll).
    * Random partial stretching in time and frequency (warping).
    * Addition of noise from samples that were rejected during preprocessing (non-salient chunk in training data) and hence might contains soundscape, or overlapping bird calls, etc.
* (**Noise injection**) The **randomly weighted addition of ambient noise** extracted from audio chunks that do **not contains bird vocalizations** was the most powerful augmentation.
* They find non-salient chunk using a detector based on signal strength since focal recordings are high SNR, so low signal implies presence of ambient noise.


## Voice of Birds: Sound of 114 species of birds till 2022 (Kaggle):
* Kaggle link: https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022
* Scraped from xeno-canto like BirdSong Denoising.
* We want to test our method of noise injection in low-noise settings.


## UrbanSound8k
* A subset of the Urban Sound dataset by J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” in Proceedings of the 22nd ACM international conference on Multimedia, Orlando Florida USA: ACM, Nov. 2014, pp. 1041–1044. doi: 10.1145/2647868.2655045.
* The objective is to define a **common vocabulary** or semantic groups of sound-source pairs for comparison across works.
* The dataset is built around the most common sound types & sources responsible for noise complaints filed in NYC from 2010 to 2014.
* **Mel Spectrogram**: A frequency-time image obtained after Fourier transform on audio signal. It shows how the frequency content of audio changes over time. Frequency is converted to mel-scale which is based on how humans hear and provides better resolution for lower frequencies than a normal spectrogram.
* **8732 labeled** Mel spectrogram images with **10 classes**. **8.75** hrs of audio.
* Urban sound classes like air conditioner, street music, car horn, children playing, siren, etc. The sounds belong to **events with different duration**, i.e., gun shot is ideally sporadic & short (1-2s) while jackhammer sound is continuous & over 30s.
* Data is split into **10 folds** to ensure standard comparison across works.
* **Metadata** for each sound excerpt in a **CSV** file like classID (numeric, 0-9).
* Freesound ID: whole recording, occurrenceID(numeric): identify diff occurrences of a sound within original recording, slideID(numeric): excerpt/slice taken from the same occurrence of a sound.
* 1 row/sample = 1 slice. Max 1000 slices per class. Each slice is max 4 seconds and for longer occurrences, they use sliding window with hop size of 2s (overlap).
* Max duration of sound restricted to 4 secs because prev work shows it's enough for human subjects.
* Salience (foreground or background) sound is critical factor in the experiment.
* Window of 23.2ms & 50% frame overlap. 40 Mel bands btw 0 to 22050 Hz.
* Original sound download link: https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz
* Mel spectrogram download link: https://www.kaggle.com/datasets/pranked03/urbansound8k-mel-spectrogram-images

## Data Setup for UrbanSound8k:
**UrbanSound8k**: This dataset contains Mel-spectrogram images of urban landscape audio.  
Dataset description: [https://www.kaggle.com/datasets/pranked03/urbansound8k-mel-spectrogram-images](https://www.kaggle.com/datasets/pranked03/urbansound8k-mel-spectrogram-images)  
Bash commands (please update the output path as per your environment):

```bash
mkdir -p ~/jobtmp/UrbanSound8k
curl -L -o ~/jobtmp/UrbanSound8k/urbansound8k-mel-spectrogram-images.zip\
  https://www.kaggle.com/api/v1/datasets/download/pranked03/urbansound8k-mel-spectrogram-images
unzip ~/jobtmp/UrbanSound8k/urbansound8k-mel-spectrogram-images.zip -d images
```

## BirdSong Denoising:
* This dataset is a subset from Xeno-Canto, a large open wildlife sound dataset.
* It involves recordings with different levels of noise severity (5 levels, A to E, with A as best and E as worst).
* Noise is mainly soundscapes (other animals or birds in the background) or ambient noises like waterfall.
* It contains recording which are stereo (left and right) which we've converted to mono audio by averaging over the channels.
* The dataset was originally meant for a denoising benchmark and has a lot of research built on top of it with people proposing different noise models and techniques that work on the dataset.
* The noise models built on this dataset are suitable to be used in SSL methods to measure the influence of noise on SSL methods.
* Further, we have the ground-truth clean audio and the raw noisy audio (with levels A to E). So we can interpolate between these two types of audio.
* Valid set: 1398 audio of varying length (1sec, 9secs, etc.).
* Train set: 10,000 original single channel audio but only 9855 unique audios (excluding repetition due to stereo).
* We convert the stereo to monophonic audio by sampling the left channel.
* The data doesn't have enough samples per species to be good for pretraining a classifier. It has 2,226 species in the train set, and 721 species in the validation set (hopefully the same species and not new ones). This can serve as a good **downstream** task because of the variety of noise in the data, and the availability of the clean data.


## BirdSong (Shuffle and Learn, Carson Range, Sierra Mountains):
* Unlabeled.
* Birds: "Quail, Blue Jays, Black Headed Grosbeaks, Doves, Robins, Red Finches, Stellars Jays, Black-billed Magpies, Yellow Warblers and Varied Thrush, among others".
* Serves as a pitch rich pretraining dataset.
* Focus is on pitch and timbre as measured by freq domain changes over time.
* Why pitch? Welders/industry know the quality of the weld based on the distance of welding tool and hence the pitch.
* Pitch & timbre are usually present in the high freq range.
* Constant-Q Transform (CQT) helps w/t high resolution of time in high freq and high freq resolution in low freq which is good for pitch & timbre extraction.

* Pretext task: Temporal order verification just like Shuffle and Learn.
* (a,b,c) v/s [(d,a,b), (b,a,d)], i.e., positive & negative pair for (a,b,c,d) in contrastive loss.
* (a,d,b), (a,b,d) and (b,d,a), (d,b,a) have some order information in them and hence possibly not a good negative sample but no such data collected by the authors.
* Learning temporal order helps learn the pitch information.
* Downstream task: Boat motor classification but paper says it might've been too easy. 2.6s i/p & 2 classes.

* 1,252 total clean, high quality unlabeled samples of diff types of birds chirping while bird feeding.
* 11 sec audio-video (44.1 kHz). Resample audio to 22 kHz sample rate.
* Split into 4 sequence chunks of 2.6s each: (a,b,c,d)
* Measure of signal 1/Cleaning: Difference between avg amd max magnitude of audio signal. Remove wind background or low signal samples.
* Measure of signal 2/Cleaning: Unsupervised K-means cluster on constant-Q transform unrolled vectors. Remove unwanted clusters of noise and low freq.
* The authors leave it to us to decide what to remove after clustering. We've removed all clusters with less than 100 samples. Most of them contained noise like engine noise, power tool noise, etc.
* Test: Distribution of samples across 8 clusters: Counter({np.int32(3): 278, np.int32(0): 238, np.int32(7): 22, np.int32(2): 1, np.int32(5): 1, np.int32(1): 1, np.int32(6): 1, np.int32(4): 1}).
* Train: CQT: Distribution of samples across clusters: Counter({np.int32(0): 785, np.int32(1): 130, np.int32(7): 4, np.int32(6): 2, np.int32(4): 1, np.int32(2): 1, np.int32(5): 1, np.int32(3): 1}).
* We were left with 1431 samples as compared to the paper's 1252 samples due to the subjective cluster selection process not specified by the authors.

* Behavior Bias: bird behavior is biased towards feeding and not really free-play.
* Sensor Bias: Even wind activated the capture and had to be removed manually which might indicate that the sensory was overly sensitive.
* Good news: Background noise is of urban objects like cars so Urban Sound 8k might be apt for background noise injection.

* Representation: 2D Constant-Q Transform (CQT) spectrogram on audio waveform input.
* (Augmentation) Pitch shifting (while keeping time same). Done in increments of semitones (Ex: A to A#).
* (Aug) Octave shifting: 12 semitones shift in audio waveform while keeping time same.
* (Aug) Time Stretching: Extend or compress the duration of the signal by reducing or increasing the sampling rate respectively.
* (Aug) SpecAugment: Mask both freq and time in the CQT spectrogram. 1 mask for freq [f, f+30). 2 masks for time [t, t+20] and [t_, t_+20], no overlap. (Idea: could be used for reconstruction pre-text?)

* Triplet Siamese network with reduced last dense layer AlexNet.
* Lecun normal initializer, leaky ReLU and a ton of drop-out.
* Only 1 of the Siamese triplets was picked for downstream and partially fine-tuned for 20 epochs.


## General jargon related to audio data:
* **Pitch**: Directly related to the fundamental frequency of a sound wave. Higher frequency = higher pitch. Measured in Hertz (Hz).
* **Timbre**: Represents the "color" or quality of a sound, allowing us to differentiate between instruments even when playing the same note. Determined by the presence and relative strength of harmonic frequencies beyond the fundamental. Described using terms like "bright," "warm," "round," or "brassy".
* **Sampling rate**: Human highest hearing frequency is 8Hz, so by Nyquist theorem, **16Hz** is enough for sampling. Training data is generally 16Hz. CD-audio is 44kHz, while high-resolution audio is 192kHz.
* **Resampling**: **Transformers** can be sensitive to sampling rate of training data so we'll need to resample during preprocessing.
* **Aliases/Downsampling**: If downsampling, make sure to filter and remove the higher frequencies before reducing to the new sampling rate & the new Nyquist theorem given freq threshold.
* Amplitude: Measured in **decibels (dB)**. Each sample of sound consists of amplitude of audio wave at that point in time.
* Audio signals: Normal speaking voice is under **60dB**, rock concert is 125dB. Real-word audio starts at **0dB**, i.e., quietest possible sound humans can hear.
* **Bit depth**: The bit depth gives the **precision** of amplitude value representation and hence the sampling precision.
* Common values: **16-bit and 24-bit** use integer representation of amplitude samples. **32-bit** uses normalized floating point representation of amplitude samples **[-1.0, 1.0] range**.
* More precision, lower the sampling noise (information loss due to quantization).
* Human **hearing is logarithmic** in nature, i.e., sensitive to small fluctuations in quiet sounds more than in loud sounds. Amplitude/loudness when described in **decibels (dB) is easy to interpret** as it's also logarithmic.
* Digital audio: **0 dB is loudest** possible amplitude and rest are in negative. **-60 dB is min** threshold below which inaudible. Every **-6 dB** is **halving** of the amplitude.
* **MFCC Baseline**: Mel-Frequency Cepstral Coefficients can be used as a competitive baseline for benchmarking.
* **Note**: Chars that represent a freq/pitch. A (440Hz), B, C, D, E, F, G.
* **Tone**: Interval between two notes when there is a note in between, for ex: going from A to B.
* A,B,C,D,E,F,G,A: (A-B) Tone, (B-C) Semitone, (C-D) Tone, (D-E) Tone, (E-F) Semitone, (F-G) Tone, (G-A) Tone. 12 semitones/pitch shifts in total.
* **Semitone**: Interval between A to A#. 1 whole tone = 2 semitones.
* **Octave**: 12 semitones combine to form 1 octave, it involves 8 Notes (One note back to that note being played again).


## Audio visualization:
* **Amplitude-Time/Time Domain**: Use **librosa** for **waveform visualization** (it will already have amplitude in the [-1.0, 1.0] range).
* **Amplitude-Frequency/Frequency Domain**: Frequencies at a frozen snapshot, i.e., **region-wise** instead of whole sound. Frequency spectrum using discrete fourier transform (DFT). Freq spectrum gives the same information as waveform.
* **DFT/FFT**: DFT is implemented as FFT for efficiency reasons. DFT gives complex no. output which we can take the abs/magnitude of to get amplitude information.
* Plotting the angle btw real and imaginary components gives us the **phase spectrum**.
* **Power Spectrum**: Do above but instead of amplitude calculate energy, i.e., the **squared amplitude**.
* **Spectrogram/Amplitude-Frequency-Time**: Measure of how frequencies change over time as measured by the amplitude corresponding to each frequency signal. Stack frequency spectrums obtained by DFT.
* **Short Time Fourier Transform (STFT)**: Algo for spectrogram. Since multiple DFTs applied it outputs **complex values** and same as freq spectrum take magnitude or angle for amplitude or phase information. For each (time,frequency) value there will be a cmap value corresponding to **amplitude or power both in dB** since dB is a ratio relative to some standard (np.max since negative amplitudes used in digital for some reason) & not a unit of power/amplitude intrinsically.
* Balance between **frequency v/s time resolution**: **Window size** determines this. If long window, then low time resolution but more frequency resolution and vice versa.
* **Inverse STFT**: Convert spectrogram into waveform. Requires the **phase information** which is lost when we do absolute of the DFT o/p. **Phase reconstruction algorithm** like "classic **Griffin-Lim** algo" used for this.
* **vocoder ANN**: Does the same thing as Inverset STFT. Ex: **HiFiGAN vocoder** on mel spectrogram which is lossy representation of data.
* **Mel Spectrogram**: **Perceptually meaningful**. Standard spectrogram has frequency on a linear scale and in Hz but human hearing is **logarithmic**, i.e., logarithmically more sensitive to changes (amplitude) in lower freq than in higher freq. **Non-linear perception** of frequency in humans is captured by mel scale.
* **Mel Filterbank/Lossy op**: After STFT, **frequency spectra** are sent through a set of filters called mel filterbank to transfer to mel scale.
* **Log-Mel Spectrogram**: Strength of mel frequency components when measured in decibels.
* **Variants of Mel**: Power or amplitude. Just log or dB (log*10) or just absolute vals of STFTs. 2 diff mel scales, "htk" and "slaney".
* **Constant-Q Transform**: CQT. Solves the problem of time-frequency resolution balance. Low frequences get higher freq resolution and lower time resolution cause they're anyways low freq. High freq get higher time resolution & so are good for human audible pitch & timbre.


# References:
* Use this course to get an overview of working with audio data: [https://huggingface.co/learn/audio-course/en/chapter1/introduction](https://huggingface.co/learn/audio-course/en/chapter1/introduction)
* Use this book for quick future reference or questions: https://brianmcfee.net/dstbook-site/content/intro.html
* Use this book to learn about the jargon surround music and audio waveforms: https://languageofmusic.ca/home.html
* MUDA: Reference python package for music related data transformations: https://github.com/bmcfee/muda/tree/master
