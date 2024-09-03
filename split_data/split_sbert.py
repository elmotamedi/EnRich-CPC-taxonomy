import os
import json
import zstandard as zstd
import io
import jsonlines
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset

from skmultilearn.model_selection import iterative_train_test_split

def split_data(indices, labels_tensor_np):
    indices = indices.reshape(-1, 1)  # Ensure indices are in the correct shape for the split function
    labels_tensor_np = labels_tensor_np.reshape(-1, labels_tensor_np.shape[-1]) 
    train_idx, train_labels, remaining_idx, remaining_labels = iterative_train_test_split(
        indices, labels_tensor_np, test_size=0.2)
    
    val_idx, val_labels, test_idx, test_labels = iterative_train_test_split(
        remaining_idx, remaining_labels, test_size=0.5)
    
    print(f"Train indices shape: {train_idx.shape}")
    print(f"Validation indices shape: {val_idx.shape}")
    print(f"Test indices shape: {test_idx.shape}")
    
    return train_idx.flatten(), val_idx.flatten(), test_idx.flatten()

def create_tensor_dataset(embeddings, labels, indices, name):
    print(f"Creating {name} dataset with indices: {indices.shape}")
    embeddings_tensor = torch.tensor(embeddings[indices])
    labels_tensor = torch.tensor(labels[indices])
    
    if embeddings_tensor.shape[0] != labels_tensor.shape[0]:
        raise ValueError(f"Shape mismatch: embeddings ({embeddings_tensor.shape[0]}) and labels ({labels_tensor.shape[0]})")
    
    return TensorDataset(embeddings_tensor, labels_tensor)

if __name__ == "__main__":
    print("Loading data")
    saved_data = torch.load('D:/Users/elham/projects/EnrichData/outputs/reformatted_sbert/99/merged_data.pt')
    input_embeddings = saved_data['embeddings']
    labels_tensor = saved_data['labels_tensor']

    indices = np.arange(len(input_embeddings))
    input_embeddings_np = input_embeddings.numpy()
    labels_tensor_np = labels_tensor.numpy()

    print("Starting data splitting!")
    
    train_idx, val_idx, test_idx = split_data(indices, labels_tensor_np)
    
    print("Creating the TensorDatasets")

    train_dataset = create_tensor_dataset(input_embeddings_np, labels_tensor_np, train_idx, "train")
    print("Train dataset created!")

    val_dataset = create_tensor_dataset(input_embeddings_np, labels_tensor_np, val_idx, "val")
    print("Validation dataset created!")

    test_dataset = create_tensor_dataset(input_embeddings_np, labels_tensor_np, test_idx, "test")
    print("Test dataset created!")

    # Save the split datasets
    torch.save(train_dataset, 'D:/Users/elham/projects/EnrichData/outputs/splits_sbert99/train/train_dataset.pt')
    torch.save(val_dataset, 'D:/Users/elham/projects/EnrichData/outputs/splits_sbert99/val/val_dataset.pt')
    torch.save(test_dataset, 'D:/Users/elham/projects/EnrichData/outputs/splits_sbert99/test/test_dataset.pt')

    print("Datasets saved successfully!")
