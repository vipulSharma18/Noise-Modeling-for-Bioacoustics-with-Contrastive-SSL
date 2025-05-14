import torch
from torchaudio.prototype.functional import extend_pitch, oscillator_bank


PI = torch.pi
PI2 = 2 * torch.pi


def generate_time_varying_sawtooth_wave(
                                        fundamental_freq=344.0,
                                        duration=1,
                                        sampling_rate=22000
                                        ):
    F0 = fundamental_freq  # fundamental frequency
    DURATION = duration  # [seconds]
    SAMPLE_RATE = sampling_rate  # [Hz]

    NUM_FRAMES = int(DURATION * SAMPLE_RATE)
    fm = 10  # rate at which the frequency oscillates [Hz]
    f_dev = 0.1 * F0  # the degree of frequency oscillation [Hz]

    phase = torch.linspace(0, fm * PI2 * DURATION, NUM_FRAMES)
    freq0 = F0 + f_dev * torch.sin(phase).unsqueeze(-1)
    amp0 = torch.ones((NUM_FRAMES, 1))

    freq, amp, waveform = sawtooth_wave(
        freq0, amp0, int(SAMPLE_RATE / F0), SAMPLE_RATE
        )
    return waveform


def sawtooth_wave(freq0, amp0, num_pitches, sample_rate):
    freq = extend_pitch(freq0, num_pitches)

    mults = [-((-1) ** i) / (PI * i) for i in range(1, 1 + num_pitches)]
    amp = extend_pitch(amp0, mults)
    waveform = oscillator_bank(freq, amp, sample_rate=sample_rate)
    return freq, amp, waveform
