import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import multiprocessing as mp
import numpy as np
import librosa
import pandas as pd


print(os.uname())
t = time.localtime()
print(f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}")
num_threads = int(os.environ.get('SLURM_CPU_COUNT', os.environ.get('SLURM_CPUS_PER_TASK', 64)))
root_dir = "data/CBC2020/"


def worker_fn(threadId):
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        samples = pd.read_csv(os.path.join(root_dir, f'{split}.csv'))['sample']
        work_per_thread = len(samples) // num_threads
        start_idx = work_per_thread * threadId
        end_idx = start_idx + work_per_thread
        samples = samples[start_idx:end_idx]

        for sample in samples:
            file_path = os.path.join(split_dir, sample)
            if os.path.isfile(file_path):
                audio, _ = librosa.load(file_path, sr=32000)
                np.save(file_path.replace('.mp3', '.npy'), audio[:7*32000])



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

    splits = ['train', 'val', 'test']
    for split in splits:
        df = pd.read_csv(os.path.join(root_dir, f'{split}.csv'))
        df['sample'] = df['sample'].apply(lambda x: x.replace('.mp3', '.npy'))
        df.to_csv(os.path.join(root_dir, f'{split}.csv'), index=False)
