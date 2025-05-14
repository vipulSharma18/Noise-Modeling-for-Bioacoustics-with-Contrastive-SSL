import os
from collections import Counter
import numpy as np
import pandas as pd
from stable_ssl.utils import seed_everything
from ssl_bioacoustics.utils import filter_low_signal_audio, CQT_and_unroll
from sklearn.cluster import KMeans


def filter_audio(audio_folder):  # pragma: no cover
    if not os.path.exists(audio_folder):
        print("Directory does not exist:", audio_folder)
        return
    print("Processing directory:", audio_folder)
    files = os.listdir(audio_folder)
    print("Total audio files:", len(files))
    tgt_folder = os.path.abspath(os.path.join(audio_folder, ".."))
    print("Output directory:", tgt_folder)
    filtered_audio = []
    for sample in files:
        data = np.loadtxt(os.path.join(audio_folder, sample), delimiter=",")
        if data.shape != (228800,):
            print(f"Interesting: Other than 228800 shape detected: {sample}")
        result = filter_low_signal_audio(data, threshold=5)
        if result:
            filtered_audio.append(sample)
    print("Total audio files after filtering:", len(filtered_audio))
    tgt_file = os.path.join(tgt_folder, os.path.basename(audio_folder) + "-filtered.csv")
    np.savetxt(tgt_file, filtered_audio, delimiter=",", fmt="%s")
    print("Output file:", tgt_file)


def cqt_cluster(data_path):  # pragma: no cover
    if not os.path.exists(data_path):
        print("CQT: Directory does not exist:", data_path)
        return
    print("CQT: Processing directory:", data_path)
    files = np.loadtxt(data_path+"-filtered.csv", delimiter=",", dtype=str)
    print("CQT: Total audio files:", len(files))
    audio_list = []
    for sample in files:
        data = np.loadtxt(os.path.join(data_path, sample), delimiter=",").flatten().tolist()
        if len(data) < 228800:
            print(f"CQT: Interesting: < 228800 audio len detected: {sample}, for data {data_path}")
        audio_list.append(data)
    cqt_unrolled = CQT_and_unroll(
        audio_list,
        length=228800,
        sr=22000,
        baseline_note='C1',
        freq_bins=70,
        bins_per_octave=12,
        )
    # clustering logic
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    y_pred_kmeans = kmeans.fit_predict(cqt_unrolled)
    assert len(y_pred_kmeans) == len(files)
    print("CQT: Distribution of samples across clusters:", Counter(y_pred_kmeans))
    y_pred_kmeans = pd.DataFrame(y_pred_kmeans, columns=["cluster"])
    y_pred_kmeans["sample"] = files
    y_pred_kmeans.to_csv(data_path+"-filtered-clustered.csv", index=False)


def pick_clean(cluster_csv_path, samples_threshold=100):  # pragma: no cover
    if not os.path.exists(cluster_csv_path):
        print("Pick clean: File doesn't exist:", cluster_csv_path)
        return
    print("Pick clean: Processing file:", cluster_csv_path, "with threshold:", samples_threshold)
    df = pd.read_csv(cluster_csv_path)
    print("Pick clean: Total samples before removal of clusters:", len(df))
    cluster_counts = Counter(df["cluster"])
    print("Pick clean: Distribution of samples across clusters:", cluster_counts)
    clusters_to_remove = [k for k, v in cluster_counts.items() if v < samples_threshold]
    print("Pick clean: Removing clusters:", clusters_to_remove)
    df = df[~df["cluster"].isin(clusters_to_remove)]
    print("Pick clean: Total samples after removal of clusters:", len(df))
    df.to_csv(cluster_csv_path.replace("-filtered-clustered.csv", "-clean.csv"), index=False)


if __name__ == "__main__":  # pragma: no cover
    seed_everything(0)
    data_paths = [
        "/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/test",
        "/users/vsharm44/projects/ssl-bioacoustics/data/Birdsong/audio/raw-audio-unfiltered-22000-sample/train",
        ]

    for data_path in data_paths:
        if os.path.exists(data_path+"-filtered.csv"):
            print("Already filtered with volume of signal thresholding:", data_path)
        else:
            filter_audio(data_path)

        if os.path.exists(data_path+"-filtered-clustered.csv"):
            print("Already processed using CQT based clustering:", data_path)
        else:
            cqt_cluster(data_path)

        if os.path.exists(data_path+"-clean.csv"):
            print("CQT clustering done:", data_path)
        else:
            pick_clean(data_path+"-filtered-clustered.csv")
    print("Preprocessing pipeline for all directories done.")
