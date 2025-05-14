import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
from PIL import Image
from ssl_bioacoustics.custom_transforms.spectrograms import Spectrogram
import multiprocessing as mp
# - _target_: ssl_bioacoustics.custom_transforms.Spectrogram
#   sampling_rate: 32000
#   spectrogram: power_mel
#   convert_to_db: True
#   representation_mode: rgb
#   kwargs:
#     n_mels: 128
#     n_fft: 2048
#     hop_length: 512

num_threads = int(os.environ.get('SLURM_NPROCS', os.environ.get('SLURM_CPU_COUNT', os.environ.get('SLURM_CPUS_PER_TASK', 1))))

def worker_fn(thread_id):
    root_dir = "/users/vsharm44/projects/ssl-bioacoustics/data/CBC2020"
    splits = ["train", "val", "test"]
    spectrogram = Spectrogram(sampling_rate=32000, spectrogram="power_mel", convert_to_db=True, representation_mode="rgb", kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512})

    for split in splits:
        source_dir = os.path.join(root_dir, split)
        target_dir = os.path.join(root_dir, split+"_spectrogram")
        os.makedirs(target_dir, exist_ok=True)

        source_files = pd.read_csv(os.path.join(root_dir, split+".csv"))
        target_files = pd.read_csv(os.path.join(root_dir, split+"_spectrogram.csv"))

        assert len(source_files) == len(target_files)

        to_process = []
        for _, row in source_files.iterrows():
            spectrogram_path = os.path.join(target_dir, row["sample"].replace(".npy", ".png"))
            if not os.path.exists(spectrogram_path):
                to_process.append(row["sample"])

        total_work = len(to_process)
        print(f"Total work: {total_work}, out of {len(source_files)}")
        print(f"Thread ID: {thread_id}, work per thread: {total_work // num_threads}")
        
        bins = np.linspace(0, total_work, num_threads, dtype=int, endpoint=False)
        
        start_idx = bins[thread_id]
        end_idx = bins[thread_id+1] if (thread_id+1)<len(bins) else total_work

        source_files = to_process[max(0, start_idx):min(total_work, end_idx)]
        k = 0
        for row in source_files:
            k += 1
            x = np.load(os.path.join(source_dir, row))
            spectrogram_path = os.path.join(target_dir, row.replace(".npy", ".png"))
            if not os.path.exists(spectrogram_path):
                x = spectrogram(x)
                if not isinstance(x, Image.Image):
                    x = Image.fromarray(x)
                parent_dir = os.path.dirname(spectrogram_path)
                os.makedirs(parent_dir, exist_ok=True)
                x.save(spectrogram_path)
            if k % 10 == 0:
                print(f"Thread {thread_id}, Progress: {split}, {k}/{total_work // num_threads}")
        print(f"Finished {split}")


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

    root_dir = "/users/vsharm44/projects/ssl-bioacoustics/data/CBC2020"
    splits = ["train", "val", "test"]

    for split in splits:
        target_dir = os.path.join(root_dir, split+"_spectrogram")
        target_files = pd.read_csv(os.path.join(root_dir, split+"_spectrogram.csv"))
        for idx, row in target_files.iterrows():
            if not os.path.exists(os.path.join(target_dir, row["sample"])):
                print(f"Missing: {row['sample']}, split: {split}")
