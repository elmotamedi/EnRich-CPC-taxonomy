import os
import signal
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, ConcatDataset
from skmultilearn.model_selection import iterative_train_test_split
from multiprocessing import Pool
from functools import partial
import io
import json
import zstandard as zstd
import jsonlines
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

N_THREADS = 24


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def read_jsonl_z(file_path):
    with open(file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.stream_reader(compressed_file)
        with io.TextIOWrapper(decompressed, encoding='utf-8') as reader:
            data = [obj for obj in jsonlines.Reader(reader)]
    return data

def concatenate_labels(row):
    return [str(a) + " " + str(b) + " " + str(c) + " " + str(d) for a, b, c, d in zip(row['label0'], row['label1'], row['label2'], row['label3'])]

def process_file(file_path):
    print(f"processing {file_path}")
    # Load the data
    data = read_jsonl_z(file_path)
    df = pd.DataFrame(data)

    df['text'] = df.apply(lambda row: f"{row['title']} {row['abstract']}" if len(f"{row['title']} {row['abstract']}") > 100 else row['text'], axis=1)
    df['label'] = df.apply(concatenate_labels, axis=1)
    df = df[['text', 'label0', 'label1', 'label2', 'label3', 'label']]

    return df

def process_files(input_dir):
    files_inp = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl.z")]

    with Pool(N_THREADS, initializer=init_worker) as pool:
        try:
            dfs = pool.map(process_file, files_inp)
        except KeyboardInterrupt:
            print("Process interrupted. Terminating pool...")
            pool.terminate()
        except Exception as e:
            print(f"An error occurred: {e}")
            pool.terminate()
        finally:
            pool.close()
            pool.join()
    
    # Concatenate all DataFrames
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def get_label_stat(df, label_name):
    labels_list = df[label_name].tolist()  
    flat_list = [x for xs in labels_list for x in xs]
    possible_labels = np.unique(flat_list)
    label_counts = Counter(flat_list)
    label_count_dict = dict(label_counts)
    return label_count_dict

def split_and_save(df, stat_dir, train_dir, val_dir, test_dir, split_type, label):
    df = df.reset_index(drop=True, inplace=True)
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_size = len(df) - train_size - val_size

    if split_type == 'random_split':
        print("implement if needed.")
        # train_dataset, test_dataset = random_split(df, [train_size + val_size, test_size])
        # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    elif split_type == 'iterative_train_test_split':
        indices = df.index.values

        labels_list = df[label].tolist()  
        flat_list = [x for xs in labels_list for x in xs]
        possible_labels = np.unique(flat_list)

        mlb = MultiLabelBinarizer(classes=possible_labels)
        y = mlb.fit_transform(df[label])
        # y_df = pd.DataFrame(y, columns=mlb.classes_)
        # y_df['labels'] = y_df.apply(lambda row:s row.tolist(), axis=1)
        print("start spliting the full dataset.")
        
        train_idx, train_labels, remaining_idx, remaining_labels = iterative_train_test_split(
            indices.reshape(-1, 1), y, test_size=0.2)
            
        print("start spliting the validation split.")
        val_idx, val_labels, test_idx, test_labels = iterative_train_test_split(
            remaining_idx, remaining_labels, test_size=0.5)
        
    train_dataset = df.iloc[train_idx.flatten()]
    val_dataset = df.iloc[val_idx.flatten()]
    test_dataset = df.iloc[test_idx.flatten()]

    # Save splits
    train_dataset.to_parquet(os.path.join(train_dir, 'train_dataset.parquet'))
    val_dataset.to_parquet(os.path.join(val_dir, 'val_dataset.parquet'))
    test_dataset.to_parquet(os.path.join(test_dir, 'test_dataset.parquet'))
    
    label_count_dict = get_label_stat(df, label)
    with open(os.path.join(stat_dir, 'label_count_dict_fulldata.json'), 'w') as f:
        json.dump(label_count_dict, f, indent=4)

    label_count_dict = get_label_stat(train_dataset, label)
    with open(os.path.join(stat_dir, 'label_count_dict_train.json'), 'w') as f:
        json.dump(label_count_dict, f, indent=4)
    
    label_count_dict = get_label_stat(val_dataset, label)
    with open(os.path.join(stat_dir, 'label_count_dict_val.json'), 'w') as f:
        json.dump(label_count_dict, f, indent=4)

    label_count_dict = get_label_stat(test_dataset, label)
    with open(os.path.join(stat_dir, 'label_count_dict_test.json'), 'w') as f:
        json.dump(label_count_dict, f, indent=4)

def load_datasets(directory, suffix):
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            filepath = os.path.join(directory, filename)
            data = pd.read_parquet(filepath)
            datasets.append(data)
    return ConcatDataset(datasets) if datasets else None

def load_all_datasets(input_dirs):
    train_dataset = load_datasets(input_dirs['train'], 'train_dataset.parquet')
    val_dataset = load_datasets(input_dirs['val'], 'val_dataset.parquet')
    test_dataset = load_datasets(input_dirs['test'], 'test_dataset.parquet')

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    input_dir = 'd:/Users/elham/projects/EnrichData/outputs/deduplicated/tmp'
    train_dir = 'd:/Users/elham/projects/EnrichData/Data/splits/999/train'
    val_dir = 'd:/Users/elham/projects/EnrichData/Data/splits/999/val'
    test_dir = 'd:/Users/elham/projects/EnrichData/Data/splits/999/test'
    stat_dir = 'd:/Users/elham/projects/EnrichData/Data/splits/999'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Split type: random_split, iterative_train_test_split
    split_type = "iterative_train_test_split"

    # df = process_files(input_dir)
    # df.to_parquet(f'{input_dir}/full_dataset_999.parquet', engine='pyarrow') # Save the full DataFrame
    df = pd.read_parquet(f'{input_dir}/full_dataset_999.parquet', engine='pyarrow')
    split_and_save(df, stat_dir, train_dir, val_dir, test_dir, split_type, 'label')
