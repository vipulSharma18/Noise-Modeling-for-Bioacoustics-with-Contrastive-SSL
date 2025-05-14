import os
import time
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from check_runs import check_runs


def generate_heatmap(csv_path, heatmap_key):
    """
    Generate a heatmap from the csv job summary created by check_runs.
    """
    df = pd.read_csv(csv_path)
    grouped_df = df.groupby(['model', 'ssl_obj', 'train_data', 'noise'])

    # Iterate over each group
    display_key = heatmap_key.replace("/", "_")
    for (model, ssl_obj, train_data, noise), group in grouped_df:
        # Pivot the table to prepare for heatmap
        heatmap_data = group.pivot(
            index='data_severity',
            columns='augmentation_severity',
            values=heatmap_key
        )

        # Create the heatmap title
        title = f"{model}_{ssl_obj}_{train_data}_{noise}_{display_key}"

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={'label': f'Accuracy ({display_key})'})
        plt.title(title)
        plt.xlabel("Augmentation Severity")
        plt.ylabel("Data Severity")
        plt.savefig("heatmaps/Heatmap_" + title + '_' + str(time.strftime('%Y%m%d-%H%M%S')) + "_.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        help="Path to the multirun output directory or directly the output csv file.")
    parser.add_argument("--key",
                        default="eval/cifar10_severity_5/acc1",
                        type=str,
                        help="Key to plot the heatmap for.")
    args = parser.parse_args()
    if os.path.exists(args.input) and not os.path.isfile(args.input):
        path = check_runs(args)
        generate_heatmap(path, args.key)
    else:
        generate_heatmap(args.input, args.key)
