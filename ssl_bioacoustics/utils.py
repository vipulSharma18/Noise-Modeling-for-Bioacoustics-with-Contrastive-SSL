"""
Reusable utils.
"""
import os
from contextlib import contextmanager
from random import seed, getstate, setstate
import numpy as np
import torch
import librosa


@contextmanager
def isolate_rng(local_seed=0):
    """
    Isolate the random number generator for a block of code.
    """
    # save global state
    hash_seed = os.environ.get("PYTHONHASHSEED")
    py_state = getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
        # the individual part is ideally redundant given we're doing all
        cuda_state = torch.cuda.get_rng_state()
        cuda_all_state = torch.cuda.get_rng_state_all()

    # seed within context
    # setting the hash seed irrespective of it was set earlier or not
    os.environ["PYTHONHASHSEED"] = str(local_seed)
    seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(local_seed)
        torch.cuda.manual_seed_all(local_seed)

    yield

    # reset global state
    # hash seed was either already set or we set it to local seed
    if os.environ.get("PYTHONHASHSEED"):
        os.environ["PYTHONHASHSEED"] = hash_seed
    version, state, gauss = py_state
    setstate((version, tuple(state), gauss))
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state)
        torch.cuda.set_rng_state_all(cuda_all_state)


def dummy_scheduler_lambda(epoch):
    return 1


def filter_low_signal_audio(audio, threshold=5):  # pragma: no cover
    """
    Adapated from https://github.com/SingingData/Birdsong-Self-Supervised-Learning/blob/master/Notebooks/Preprocessing/2--Filter%20out%20low%20volume%20sample.ipynb.
    P. Ryan, S. Takafuji, C. Yang, N. Wilson, and C. McBride, “Using Self-Supervised Learning of Birdsong for  Downstream Industrial Audio Classification”.
    Filter out audio with low signal.
    Returns True if the peak of the audio is (1 + threshold)*100 percent greater than the average.
    """
    abs_audio = np.abs(audio)
    max_audio = np.amax(abs_audio)
    mean_audio = np.mean(abs_audio)
    return max_audio > mean_audio * (1 + threshold)


def CQT_and_unroll(audio_list,  # pragma: no cover
                   length=228800,
                   sr=22000,
                   baseline_note='C1',
                   freq_bins=70,
                   bins_per_octave=12
                   ):
    """
    Take as input a list of audio arrays and return a CQT transformed and unrolled array.
    """
    if type(audio_list) is not list:
        audio_list = [audio_list]
    cqt_list = []
    for audio in audio_list:
        audio_slice = np.asarray(audio[:length]).flatten()
        if len(audio_slice) < length:
            # Tile and trim to the required length
            audio_slice = np.tile(audio_slice, (length // len(audio_slice) + 1))[:length]
        assert len(audio_slice) == length
        constant_q = np.abs(
            librosa.cqt(
                audio_slice,
                sr=sr,
                fmin=librosa.note_to_hz(baseline_note),
                n_bins=freq_bins,
                bins_per_octave=bins_per_octave,
                )
            )
        constant_q_shape_1 = constant_q.shape[1]
        constant_q_shape_0 = constant_q.shape[0]
        cqt_list.append(constant_q)
    cqt_array = np.array(cqt_list).reshape(-1, constant_q_shape_0, constant_q_shape_1)
    # print("CQT transformed audio shape:", cqt_array.shape)
    cqt_unrolled = cqt_array.reshape(cqt_array.shape[0], -1)
    # print("Unrolled CQT transformed audio shape:", cqt_unrolled.shape)
    return cqt_unrolled
