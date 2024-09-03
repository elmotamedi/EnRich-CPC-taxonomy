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
import pickle

from accelerate import Accelerator, DistributedType

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].squeeze()
        attention_mask = encoded_dict['attention_mask'].squeeze()
        token_type_ids = encoded_dict['token_type_ids'].squeeze()
        labels = torch.tensor(label, dtype=torch.float)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': labels}

class BERTClass(torch.nn.Module):
    def __init__(self, num_labels, trained_model_path):
        super(BERTClass, self).__init__()

        MODEL_PATH = trained_model_path
        self.config = AutoConfig.from_pretrained(f"{MODEL_PATH}/config.json")
        self.bert_model = AutoModel.from_pretrained(MODEL_PATH, config=self.config)

        hidden_size = self.config.hidden_size
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        cls_output = output.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        output_dropout = self.dropout(cls_output)
        output = self.linear(output_dropout)
        return output

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
    token_type_ids = []

    mlb = MultiLabelBinarizer(classes=possible_labels)
    transformed_labels = mlb.fit_transform(data[label])

    for text in data['text']:

        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels_tensor =  torch.tensor(transformed_labels, dtype=torch.float)
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels_tensor)
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

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    with open(os.path.join(trained_model_path, 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)
    
    train_labels = mlb.fit_transform(train_df[label].tolist())
    val_labels = mlb.transform(val_df[label].tolist())
    test_labels = mlb.transform(test_df[label].tolist())

    tokenizer = AutoTokenizer.from_pretrained(model_name) 

    # Prepare dataset and dataloader
    test_dataset = CustomDataset(test_df['text'].tolist(), test_labels.tolist(), tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    ### load trained model
    num_labels = len(mlb.classes_)
    MODEL_PATH = trained_model_path
    model =  BERTClass(num_labels, MODEL_PATH)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'pytorch_model.bin')))

    model.to(device)
    model.eval()

    print('Testing started...')
    total_test_accuracy = 0
    total_test_f1_micro = 0
    total_test_f1_macro = 0

    testing_stats = []

    for step, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.float)

      
        outputs = model(input_ids, attention_mask, token_type_ids)
        logits = outputs
        logits = torch.sigmoid(logits).cpu().detach().numpy().round()
        label_ids = labels.to('cpu').numpy()
        
        total_test_f1_micro += compute_f1_micro(label_ids, logits)
        total_test_f1_macro += compute_f1_macro(label_ids, logits)
        total_test_accuracy += flat_accuracy(label_ids, logits)
        
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
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    trained_model_path = 'D:/Elham/EnrichMyData/EnrichData_PC/model/sbert_model/999/label'
    output_stat_path = "D:/Elham/EnrichMyData/EnrichData_PC/model/sbert_model/999/label/stats/testing_stats.json"
    main(train_file, val_file, test_file, model_name, 'label', trained_model_path, output_stat_path)