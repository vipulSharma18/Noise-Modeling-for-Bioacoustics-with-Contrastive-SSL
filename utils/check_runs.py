import os
import subprocess
import time
import argparse
from pathlib import Path

import jsonlines
import omegaconf

import pandas as pd
import matplotlib.pyplot as plt


def jsonl_run(path):
    """Load config and values from a single run directory."""
    _path = Path(path)
    if not _path.is_dir():
        raise ValueError(f"The provided path ({path}) is not a directory!")
    # load the config
    if not (_path / "hparams.yaml").is_file():
        raise ValueError(
            f"The provided path ({path}) must at least contain a `hparams.yaml` file."
        )
    config = omegaconf.OmegaConf.load(_path / "hparams.yaml")
    values = []
    # load the values
    if (_path / "logs_rank_0.jsonl").is_file():
        for obj in jsonlines.open(_path / "logs_rank_0.jsonl").iter(
            type=dict, skip_invalid=True
        ):
            values.append(obj)
    return config, values


def check_runs(args):
    df_dict = []
    input = os.path.expanduser(args.input)
    print("Running on args:", input)
    total = len(os.listdir(input))
    progress = 0
    parent_job_id = [
        d
        for d in os.listdir(input + "/.submitit")
        if os.path.isdir(input + f"/.submitit/{d}")
    ][0].split("_")[0]
    print("Parent job id:", parent_job_id)
    for run in os.listdir(input):
        if not os.path.isdir(os.path.join(input, run)) or run[0] == ".":
            continue

        run_path = os.path.join(input, run, "logs")

        config, metrics = jsonl_run(run_path)
        noise = config["corruption_type"]
        data_severity = config["data_noise"]
        augmentation_severity = config["augmentation_noise"]
        final_model = config["log"]["final_model_name"]
        model = config["model"]["backbone_model"]
        ssl_obj = config["model"]["name"]
        train_data = config["data"]["datasets"][config["data"]["train_on"]]["name"]
        total_epochs = config["optim"]["epochs"]

        if len(metrics) == 0:
            last_epoch = -1  # this signifies that nothing was run, not even epoch 0
        elif "epoch" in metrics[-1]:
            last_epoch = metrics[-1]["epoch"]
        else:
            found = False
            idx = -1
            while not found:
                idx -= 1
                if "epoch" in metrics[idx]:
                    last_epoch = metrics[idx]["epoch"]
                    found = True

        err_log = os.path.join(
            input, f".submitit/{parent_job_id}/{parent_job_id}_0_log.err"
        )
        out_log = os.path.join(
            input, f".submitit/{parent_job_id}/{parent_job_id}_0_log.out"
        )
        if not os.path.exists(err_log):
            # multirun
            # run is like Fog_data=5_aug=4_model=resnet18_seed=0_job=500
            task_id = run.split("_")[-1].split("=")[-1]
            err_log = os.path.join(
                input, f".submitit/{parent_job_id}_{task_id}/{parent_job_id}_{task_id}_0_log.err"
            )
            out_log = os.path.join(
                input, f".submitit/{parent_job_id}_{task_id}/{parent_job_id}_{task_id}_0_log.out"
            )

        with open(err_log, "r") as f:
            line = [line for line in f if "Training: self.epoch=1:" in line]
            if len(line) == 0:
                runtime = 0
            else:
                runtime = line[-1].split(" ")[4][2:6]
                runtime = int(runtime.split(":")[0]) * 60 + int(runtime.split(":")[1])

        error = subprocess.run("grep -i 'Error' " + out_log, shell=True, capture_output=True).stdout.decode("utf-8")

        # scrape the log for GPU ID and hostname
        hostname = subprocess.run("grep -i 'hostname=' " + out_log, shell=True, capture_output=True).stdout.decode("utf-8")
        hostname = hostname[hostname.find("hostname="):]
        hostname = hostname[:hostname.find(",")].split("=")[1]

        gpu_id = subprocess.run("grep -i 'GPU-' " + out_log, shell=True, capture_output=True).stdout.decode("utf-8")
        gpu_id = gpu_id[gpu_id.find("GPU-"):]
        gpu_id = gpu_id[:gpu_id.find(",")]
        run_summary = {
            "run": run,
            "model": model,
            "ssl_obj": ssl_obj,
            "train_data": train_data,
            "noise": noise,
            "data_severity": data_severity,
            "augmentation_severity": augmentation_severity,
            "final_model": final_model,
            "total_epochs": total_epochs,
            "last_epoch": last_epoch,
            "runtime": runtime,
            "error_log": error,
            "hostname": hostname,
            "gpu_id": gpu_id,
        }

        # add the eval accuracy dict into run_summary
        found = False
        eval_metrics = None
        for idx in range(len(metrics)-1, -1, -1):
            found = sum("eval" in x for x in list(metrics[idx].keys())[-2:])  # check at least 2 keys since 1 might be epoch
            if found:
                eval_metrics = metrics[idx]
                break
        if eval_metrics:
            run_summary.update(eval_metrics)

        df_dict.append(run_summary)
        progress += 1
        if progress % 50 == 0:
            print("Processing run:", run_path)
            print("Progress: {}/{}, {}%".format(progress, total, 100 * progress / total))

    df = pd.DataFrame(df_dict)
    print(df.head())
    print(df.columns)
    print("Generating Histograms of last epoch.")
    print("Types of trunks:", sorted(df["model"].unique()))
    print("Types of Pretraining objectives:", df["ssl_obj"].unique())
    print("Types of noise:", sorted(df["noise"].unique()))
    print("Datasets being used:", sorted(df["train_data"].unique()))
    print("Levels of data severity:", sorted(df["data_severity"].unique()))
    print(
        "Levels of augmentation severity:", sorted(df["augmentation_severity"].unique())
    )
    print("Total failed runs:", len(df[df["last_epoch"] == -1]), "out of total:", len(df))
    print("------------------------------------------------------")
    print("Sample of errors in logs:", df["error_log"].unique()[:5])
    print("------------------------------------------------------")
    path = os.getcwd() + f"/job_summary{time.strftime('%Y%m%d-%H%M%S')}.csv"
    df.to_csv(path, index=False)
    return path


def plot_epochs_histograms(job_summary):
    df = pd.read_csv(job_summary)
    plt.figure(figsize=(18, 8))
    for corruption in df["noise"].unique():
        for ssl_obj in df["ssl_obj"].unique():
            filtered_df = df[(df["noise"] == corruption) & (df["ssl_obj"] == ssl_obj)]
            if len(filtered_df) > 0:
                plt.hist(
                    filtered_df["last_epoch"],
                    bins=30,
                    histtype="step",
                    label="{}_{}".format(corruption, ssl_obj),
                )
    models = ",".join(sorted(df["model"].unique().tolist()))
    total_epochs = ",".join(sorted(df["total_epochs"].unique().astype("str").tolist()))
    ssl_obj = ",".join(sorted(df["ssl_obj"].unique().tolist()))
    corruptions = ",".join(sorted(df["noise"].unique().tolist()))
    plt.title(
        f"Corruption: {corruptions}, SSL Objective: {ssl_obj}, Model: {models}, Total epochs:{total_epochs}"
    )
    plt.xlabel("Last Epoch")
    plt.ylabel("Frequency")
    plt.legend()
    path = os.getcwd() + f"/job_summary_histograms{time.strftime('%Y%m%d-%H%M%S')}.png"
    plt.savefig(path)

    plt.figure()
    # Count occurrences of each unique value in 'last_epoch'
    counter = df["last_epoch"].value_counts().sort_index()

    # Plot as a bar chart
    ax = counter.plot(kind="bar", width=0.8)
    plt.xlabel("Last Epoch")
    plt.ylabel("Count")
    plt.title("Counts of Each Last Epoch, Total: {}".format(len(df)))
    # Add annotations
    for index, value in enumerate(counter):
        ax.text(index, value, str(value), ha="center", va="top")
    path = os.getcwd() + f"/job_summary_epoch_barchart{time.strftime('%Y%m%d-%H%M%S')}.png"
    plt.savefig(path)


def plot_runtime_boxplot(job_summary):
    df = pd.read_csv(job_summary)
    models = ",".join(sorted(df["model"].unique().tolist()))
    ssl_obj = ",".join(sorted(df["ssl_obj"].unique().tolist()))
    corruptions = ",".join(sorted(df["noise"].unique().tolist()))
    print("Generating Boxplot of runtime.")

    plt.figure(figsize=(18, 8))
    for i, corruption in enumerate(corruptions.split(",")):
        filtered_df = df[df["noise"] == corruption]
        if len(filtered_df) > 0:
            plt.boxplot(
                filtered_df["runtime"],
                positions=[i],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                label=corruption,
            )
    plt.xticks(range(len(corruptions.split(","))), corruptions.split(","))
    plt.title(f"Corruption: {corruptions}, SSL Objective: {ssl_obj}, Model: {models}")
    plt.xlabel("Corruptions")
    plt.ylabel("Runtime (s)")
    path = os.getcwd() + f"/job_summary_runtime{time.strftime('%Y%m%d-%H%M%S')}.png"
    plt.savefig(path)


def plot_summary(job_summary):
    plot_epochs_histograms(job_summary)
    plot_runtime_boxplot(job_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, help="Function to run")
    parser.add_argument(
        "--input", type=str, help="Path to the multirun output directory"
    )
    parser.add_argument("--num_workers", type=str, help="Parallel workers to use")
    args = parser.parse_args()
    if args.function == "check_runs":
        path = check_runs(args)
        plot_summary(path)
    elif args.function == "plot_epochs_histograms":
        plot_epochs_histograms(args.input)
    elif args.function == "plot_runtime_boxplot":
        plot_runtime_boxplot(args.input)
