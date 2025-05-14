import pandas as pd
import os

cwd = os.getcwd()
prj_dir = cwd[:cwd.index("ssl-bioacoustics")+len("ssl-bioacoustics")]

for split in ["all_audio", "train", "val", "test"]:
    print("-"*100)
    print(f"Verifying {split} split")
    root_dir = f"{prj_dir}/data/CBC2020/{split}/"
    species = []
    for d in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, d)):
            species.append(d)
    species_samples = {}
    for s in species:
        species_samples[s] = len(os.listdir(os.path.join(root_dir, s)))

    # Get value counts from dataframe
    all_df = pd.read_csv(f"{prj_dir}/data/CBC2020/{split}.csv")
    df_counts = all_df['ebird_code'].value_counts()

    # Check that species_samples and df_counts have same keys and values
    all_match = True
    for species, count in species_samples.items():
        if species not in df_counts.index:
            print(f"Species {species} in species_samples but not in dataframe")
            all_match = False
        elif count != df_counts[species]:
            print(f"Mismatch for {species}: os: {count} vs df: {df_counts[species]}")
            all_match = False

    if all_match:
        print(split, ": All species counts match between files and dataframe")
    else:
        print(split, ": Found mismatches between files and dataframe")
