import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
from PIL import Image
from ssl_bioacoustics.custom_transforms.spectrograms import Spectrogram
import librosa
import torchaudio
import torch
import multiprocessing as mp


num_threads = int(os.environ.get('SLURM_NPROCS', os.environ.get('SLURM_CPU_COUNT', os.environ.get('SLURM_CPUS_PER_TASK', 1))))

def worker_fn(thread_id):
    root_dir = "/users/vsharm44/projects/ssl-bioacoustics/data/CBC2020"
    splits = ["noise"]
    # spectrogram = Spectrogram(sampling_rate=32000, spectrogram="power_mel", convert_to_db=True, representation_mode="rgb", kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512})

    for split in splits:
        source_dir = os.path.join(root_dir, split)

        source_files = pd.read_csv(os.path.join(root_dir, split+".csv"))
        total_work = len(source_files)
        print(f"Total work: {total_work}")
        bins = np.linspace(0, total_work, num_threads, dtype=int, endpoint=False)
        start_idx = bins[thread_id]
        end_idx = bins[thread_id+1] if (thread_id+1)<len(bins) else total_work
        source_files = source_files[start_idx:end_idx]
        print(f"Thread {thread_id}, work per thread: {start_idx} - {end_idx}, start_idx: {start_idx}, end_idx: {end_idx}")

        to_be_removed = 0
        total = 0
        for folder in os.listdir(source_dir):
            for file in os.listdir(os.path.join(source_dir, folder)):
                total += 1
                file_xc_id = file[2:]
                file_xc_id = int(file_xc_id[:-4])
                if not file_xc_id in source_files["xc_id"].values:
                    print(f"Need to remove {file_xc_id}")
                    to_be_removed += 1
                    # os.remove(os.path.join(source_dir, folder, file))
        left = total - to_be_removed
        print(f"Finished {split}, {to_be_removed} files to be removed, total {total}, left {left}, lenght of source_files {len(source_files)}")


def run_pooled():
    print(f"Using {num_threads} threads")
    # Create process pool
    pool = mp.Pool(processes=num_threads)

    try:
        # Map worker function across thread IDs 0-63
        results = pool.map(worker_fn, range(num_threads))
    finally:
        pool.close()
        pool.join()

    return results


if __name__ == '__main__':
    run_pooled()
