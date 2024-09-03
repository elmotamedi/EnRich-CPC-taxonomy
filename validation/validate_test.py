import os
import numpy as np
from collections import Counter
import json

import torch
from datasets import load_dataset, load_metric, concatenate_datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler,TensorDataset, random_split, ConcatDataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    DistilBertTokenizer, 
    DistilBertModel,
    get_scheduler,
    set_seed
)
from torch.optim import AdamW
import datasets


from accelerate import Accelerator, DistributedType

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_probs(logits, threshold=0.5):
    sigm = 1 / (1 + np.exp(-logits))
    return sigm >= threshold

def flat_accuracy(preds, labels):
    res = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        res[i] = np.all(preds[i] == labels[i])
    return np.sum(res) / labels.shape[0]

def compute_f1_macro(out, pred):
    return metrics.f1_score(pred, out, average='macro', zero_division=1)

def compute_f1_micro(out, pred):
    return metrics.f1_score(pred, out, average='micro', zero_division=1)

def tokenize_function(data, label, possible_labels, tokenizer):
    input_ids = []
    attention_masks = []

    mlb = MultiLabelBinarizer(classes=possible_labels)
    transformed_labels = mlb.fit_transform(data[label])

    for text in data['text']:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=500,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensor = torch.tensor(transformed_labels)
    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    return dataset

def get_possible_labels(train_file, val_file, test_file, tokenizer, label, stat_file):
    train_dataset = load_dataset('parquet', data_files={'train': train_file})
    val_dataset = load_dataset('parquet', data_files={'validation': val_file})
    test_dataset = load_dataset('parquet', data_files={'test': test_file}).shuffle(seed=42)


    labels_list = train_dataset['train'][label]
    flat_list = [x for xs in labels_list for x in xs]
    train_possible_labels = np.unique(flat_list)
    n_labels_train = len(train_possible_labels)

    labels_list = val_dataset['validation'][label]
    flat_list = [x for xs in labels_list for x in xs]
    val_possible_labels = np.unique(flat_list)
    n_labels_val = len(val_possible_labels)

    labels_list = test_dataset['test'][label]
    flat_list = [x for xs in labels_list for x in xs]
    test_possible_labels = np.unique(flat_list)
    n_labels_test = len(test_possible_labels)

    i, n_labels = max(enumerate([n_labels_train, n_labels_val, n_labels_test]), key=lambda x: x[1])

    if i == 0:
        possible_labels = train_possible_labels
    elif i ==  1:
        possible_labels = val_possible_labels
    elif i == 2:
        possible_labels = test_possible_labels

    serializable_possible_labels = possible_labels.tolist()
    label_stat = {
    'num_labels': n_labels,
    'possible_labels': serializable_possible_labels
    }
    with open(stat_file, 'w') as f:
        json.dump(label_stat, f, indent=4)
    return label_stat

def main(train_file, val_file, test_file, model_name, label, trained_model_path, output_stat_path):
    #### define varialbes ###
    seed = 42
    batch_size = 16
    ############
    set_seed(seed)
    train_dataset = load_dataset('parquet', data_files={'train': train_file})
    val_dataset = load_dataset('parquet', data_files={'validation': val_file})
    test_dataset = load_dataset('parquet', data_files={'test': test_file}) 
    test_dataset = load_dataset('parquet', data_files={'test': test_file}).shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name) 

    labels_list = train_dataset['train'][label]
    flat_list = [x for xs in labels_list for x in xs]
    train_possible_labels = np.unique(flat_list)
    n_labels_train = len(train_possible_labels)

    labels_list = val_dataset['validation'][label]
    flat_list = [x for xs in labels_list for x in xs]
    val_possible_labels = np.unique(flat_list)
    n_labels_val = len(val_possible_labels)

    labels_list = test_dataset['test'][label]
    flat_list = [x for xs in labels_list for x in xs]
    test_possible_labels = np.unique(flat_list)
    n_labels_test = len(test_possible_labels)

    i, n_labels = max(enumerate([n_labels_train, n_labels_val, n_labels_test]), key=lambda x: x[1])

    if i == 0:
        possible_labels = train_possible_labels
    elif i ==  1:
        possible_labels = val_possible_labels
    elif i == 2:
        possible_labels = test_possible_labels

    ### load trained model
    MODEL_PATH = trained_model_path
    config = AutoConfig.from_pretrained(f"{MODEL_PATH}/config.json")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config = config)

    model.to(device)
    # make sure token embedding weights are still tied if needed
    model.tie_weights()
    model.eval()

    test_dataset = tokenize_function(test_dataset['test'], label, possible_labels, tokenizer) 
    test_dataloader = DataLoader(
            test_dataset, 
            sampler = RandomSampler(test_dataset), 
            batch_size = batch_size)

    print('Testing started...')
    total_test_accuracy = 0
    total_test_f1_micro = 0
    total_test_f1_macro = 0

    testing_stats = []

    for step, batch in enumerate(test_dataloader):
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device)
        batch_labels = batch[2].float().to(device)

        with torch.no_grad():        
            outputs = model(batch_input_ids, 
                            attention_mask=batch_input_mask,
                            labels=batch_labels)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        
        total_test_f1_micro += compute_f1_micro(get_probs(logits), label_ids)
        total_test_f1_macro += compute_f1_macro(get_probs(logits), label_ids)
        total_test_accuracy += flat_accuracy(get_probs(logits), label_ids)
        
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    avg_test_f1_micro = total_test_f1_micro / len(test_dataloader)
    avg_test_f1_macro = total_test_f1_macro / len(test_dataloader)
    testing_stats.append(
            {
                'test_accuracy': avg_test_accuracy,
                'test_f1_micro': avg_test_f1_micro,
                'test_f1_macro': avg_test_f1_macro
            })

    print('Test metrics:')
    print('----------------------')
    print(f'Accuracy : {avg_test_accuracy:.4f}')
    print(f'f1-score micro : {avg_test_f1_micro:.4f}')
    print(f'f1-score macro : {avg_test_f1_macro:.4f}')

    output_file = output_stat_path
    print(testing_stats)
    with open(output_file, 'w') as f:
        json.dump(testing_stats, f, indent=4)
    print('Training finished...')

    print()
    print("Testing finished...")





if __name__ == "__main__":
    train_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/999/train/train_dataset.parquet'
    val_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/999/val/val_dataset.parquet'
    test_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/999/test/test_dataset.parquet'
    model_name = 'distilroberta-base'
    trained_model_path = 'D:/Elham/EnrichMyData/EnrichData_PC/model/distilroberta_model/999/label'
    output_stat_path = "D:/Elham/EnrichMyData/EnrichData_PC/model/distilroberta_stats/999/testing_stats.json"
    main(train_file, val_file, test_file, model_name, 'label', trained_model_path, output_stat_path)