import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd

import torch
import torchvision

from ssl_bioacoustics.custom_datasets import AudioDataset

from ssl_bioacoustics.custom_transforms import (
    PreprocessSlidingWindowWithLabels,
    PreprocessSlidingWindowMetadata,
    Spectrogram,
)

import multiprocessing as mp



num_threads = int(os.environ.get('SLURM_CPU_COUNT', os.environ.get('SLURM_CPUS_PER_TASK', 64)))

def worker_fn(thread_id):
    """
    Function to be run in each worker thread
    Args:
        thread_id: Integer ID of the thread (0-63)
    """

    sliced_dataset = AudioDataset(
        root_dir="/users/vsharm44/projects/ssl-bioacoustics/data/CBC2020/",
        meta_file="/users/vsharm44/projects/ssl-bioacoustics/data/CBC2020/train.csv",
        split="train",
        sampling_rate=32000,
        transform=torchvision.transforms.v2.Compose([
            Spectrogram(
                sampling_rate=32000,
                representation="power_mel",
                convert_to_db=True,
                representation_mode="grayscale",
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                ),
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.CenterCrop(size=224),
            torchvision.transforms.v2.ToDtype(dtype=torch.float32, scale=True),
            torchvision.transforms.v2.Lambda(lambda x: torch.flatten(x)),
            ]),
        length=7,
        preprocess_fn=PreprocessSlidingWindowWithLabels(
            window_size=7, hop_length=7, sampling_rate=32000),
        preprocess_metadata_fn=PreprocessSlidingWindowMetadata(
            window_size=7, hop_length=7, individual_length_column_name="duration"),
        preprocessed_cache_dir="/users/vsharm44/projects/ssl-bioacoustics/data/CBC2020/preprocessed_cache",
        reuse_preprocessed_data=True,
        )

    nan_indices = sliced_dataset.meta_data[sliced_dataset.meta_data['preprocessed_location'].isna()].index.tolist()
    per_thread_work = len(nan_indices) // num_threads

    start_idx = thread_id * per_thread_work
    end_idx = start_idx + per_thread_work

    for idx in range(start_idx, end_idx):
        data_idx = nan_indices[idx]
        if((idx-start_idx)%500 == 0):
            sliced_dataset.meta_data['preprocessed_location'] = sliced_dataset.set_preprocessed_locations(sliced_dataset.meta_data)
            print('thread_id:', thread_id, 'running:', data_idx, 'progress:', f'{(idx-start_idx)/(end_idx-start_idx):.2%}')
            print('already preprocessed data:', len(sliced_dataset.meta_data['preprocessed_location']) - sliced_dataset.meta_data['preprocessed_location'].isna().sum())
        try:
            x, y = sliced_dataset[data_idx]
        except Exception as e:
            print('error:', e)
            print('data idx:', data_idx, 'thread_id:', thread_id)



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
