import os
import json
import time
import requests
import pandas as pd
import argparse


def create_query_ids_from_csv(data_path, file_col='xc_id', output_file="query_ids.csv"):
    # output is a csv file with the api query list
    if not os.path.exists(data_path):
        raise ValueError("Path does not exist")
    if not os.path.exists(output_file):
        print(
            f"{output_file} couldn't be found! Ignoring existing queried data."
            )
        existing_data = pd.DataFrame({'ids': [], 'response': []})
    else:
        existing_data = pd.read_csv(output_file)
    requests_ids = pd.read_csv(data_path)
    if file_col not in requests_ids.columns:
        raise ValueError("Column not found in the csv file")
    requests_ids = requests_ids[file_col].astype(str).tolist()
    requests_ids = [
        i.split('XC')[-1] for i in requests_ids  # remove the XC at the start of id.
        if i not in existing_data['ids'].astype(str).tolist()
        ]
    requests_ids = list(set(requests_ids))
    return requests_ids


def create_query_ids_from_path(data_path, output_file="query_ids.csv"):
    # output is a csv file with the api query list
    if not os.path.exists(data_path):
        raise ValueError("Path does not exist")
    if not os.path.exists(output_file):
        print(
            f"{output_file} couldn't be found! Ignoring existing queried data."
            )
        existing_data = pd.DataFrame({'ids': [], 'response': []})
    else:
        existing_data = pd.read_csv(output_file)
    requests_ids = [
        i.split('.')[0].split('_')[0][2:]
        for i in os.listdir(data_path)
        if i.endswith(".wav")
        ]
    requests_ids = [
        i for i in requests_ids
        if i not in existing_data['ids'].astype(str).tolist()
        ]
    requests_ids = list(set(requests_ids))
    return requests_ids


def get_data(query_id):
    response = requests.get(query_id)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def query_xeno_canto(requests_ids):
    data = []
    base = 'https://xeno-canto.org/api/2/recordings?query=nr:'
    for idx, i in enumerate(requests_ids):
        response = get_data(base+i)
        if response is not None:
            data.append(json.dumps(response))
        else:
            data.append('None')
            print('Failed to get data for query id:', i)
        time.sleep(1)
        if idx%100 == 0:
            print(f"Processed {idx} queries.")
    return data


def save_data(data, requests_ids, output_file='query_ids.csv'):
    if not os.path.exists(output_file):
        print(
            f"Creating a new {output_file} since couldn't find an existing one"
            )
        existing_data = pd.DataFrame({'ids': [], 'response': []})
    else:
        existing_data = pd.read_csv(output_file)
        print(
            f"Found an existing {output_file} and appending to it."
            )
    new_data = pd.DataFrame({'ids': requests_ids, 'response': data})
    new_data['ids'] = new_data['ids'].astype(str)
    existing_data['ids'] = existing_data['ids'].astype(str)
    df = pd.concat([existing_data, new_data], ignore_index=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the data folder or the csv file containing the ids',
        default='/users/vsharm44/scratch/bioacoustics_denoising/dummy'
        )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file name',
        default='dummy_query_ids.csv'
        )
    parser.add_argument(
        '--file_col',
        type=str,
        help='Column name in the csv file containing the ids',
        default='xc_id'
    )
    parser.add_argument(
        '--from_csv',
        action='store_true',
        help='Flag to indicate if the data is in a csv file'
    )
    args = parser.parse_args()

    if args.from_csv:
        requests_ids = create_query_ids_from_csv(
            args.data_path,
            file_col=args.file_col,
            output_file=args.output_file
            )
    else:
        requests_ids = create_query_ids_from_path(
            args.data_path,
            output_file=args.output_file
            )
    print(f'Requesting {len(requests_ids)} queries.')
    data = query_xeno_canto(requests_ids)
    print(f'Got data for {len(data)} queries.')
    save_data(
        data,
        requests_ids,
        output_file=args.output_file
        )
