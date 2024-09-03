import os
import signal
import numpy as np

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
from accelerate import Accelerator, DistributedType
import datasets
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn import metrics

import transformers
from datasets import load_dataset, load_metric, concatenate_datasets

from transformers import (
    AutoTokenizer,
    AutoModel,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    get_scheduler,
    set_seed
)
from torch.optim import AdamW

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def extract_emb2(model, tokenizer, batch, device):
    pad_token_id = tokenizer.pad_token_id
    with torch.no_grad():
        outputs = model(
            input_ids = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device),
            token_type_ids = batch['token_type_ids'].to(device)
        )

        x = outputs.last_hidden_state

        pad_mask = (batch['input_ids'] != pad_token_id).float()
        pad_mask = pad_mask.unsqueeze(-1).expand(x.size()).float().to(device)

        x = x * pad_mask

        mean_pooling = x[:,1:-1,:].sum(dim=1) / pad_mask[:,1:-1].sum(dim=1)
    

    return mean_pooling

def main(train_file, val_file, test_file, output_dir, model_name, label, output_stat_path, n_labels, possible_labels):
    seed = 42
    # batch_size = 16
    batch_size = 32
    weight_decay = 0.1
    # learning_rate = 1e-5
    learning_rate = 4e-5
    epochs = 5
    lr_scheduler_type = "linear" #choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    num_warmup_steps = 0
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    early_stop = True
    ###############################
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    set_seed(seed)
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    #
    #load dataset
    train_dataset = load_dataset('parquet', data_files={'train': train_file})
    val_dataset = load_dataset('parquet', data_files={'validation': val_file})
    test_dataset = load_dataset('parquet', data_files={'test': test_file}) 

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # max_len = np.zeros(len(train_dataset['train']['text']))
    # for i in range(len(train_dataset['train']['text'])):
    #     input_ids = tokenizer.encode(train_dataset['train']['text'][i], add_special_tokens=True)
    #     max_len[i] = len(input_ids)
    # print('Max length: ', max_len.max())

    # labels_list = train_dataset['train'][label]
    # flat_list = [x for xs in labels_list for x in xs]
    # train_possible_labels = np.unique(flat_list)
    # n_labels_train = len(train_possible_labels)

    # labels_list = val_dataset['validation'][label]
    # flat_list = [x for xs in labels_list for x in xs]
    # val_possible_labels = np.unique(flat_list)
    # n_labels_val = len(val_possible_labels)

    # labels_list = test_dataset['test'][label]
    # flat_list = [x for xs in labels_list for x in xs]
    # test_possible_labels = np.unique(flat_list)
    # n_labels_test = len(test_possible_labels)

    # i, n_labels = max(enumerate([n_labels_train, n_labels_val, n_labels_test]), key=lambda x: x[1])
    # if i == 0:
    #     possible_labels = train_possible_labels
    # elif i ==  1:
    #     possible_labels = val_possible_labels
    # elif i == 2:
    #     possible_labels = test_possible_labels

    class2id = {class_:id for id, class_ in enumerate(possible_labels)}
    id2class = {id:class_ for class_, id in class2id.items()}

    # model = DistilBertForSequenceClassification.from_pretrained(
    #     model_name,
    #     problem_type="multi_label_classification",
    #     num_labels = n_labels, 
    #     output_attentions = False,
    #     output_hidden_states = False, 
    # )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels = n_labels, 
        output_attentions = False,
        output_hidden_states = False, 
        id2label=id2class,
        label2id=class2id
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(model.parameters(), lr=learning_rate)


    train_dataset = tokenize_function(train_dataset['train'], label, possible_labels, tokenizer)
    val_dataset = tokenize_function(val_dataset['validation'], label, possible_labels, tokenizer)
    test_dataset = tokenize_function(test_dataset['test'], label, possible_labels, tokenizer)


    train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size)

    valid_dataloader = DataLoader(
            val_dataset, 
            sampler = RandomSampler(val_dataset), 
            batch_size = batch_size)
    
    test_dataloader = DataLoader(
            test_dataset, 
            sampler = RandomSampler(test_dataset), 
            batch_size = batch_size)
    
    
    print("done with dataloading.")
    device = accelerator.device
    print(f'accelerator device is {device}')
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, test_dataloader)
    
    max_train_steps = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler(
        name= lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps= num_warmup_steps,
        num_training_steps= max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    starting_epoch = 0
    completed_steps = 0
    lowest_valid_loss, highest_valid_acc, patience = 0, 0, 0
    break_flag1, break_flag2 = False, False
    training_stats = []

    for epoch in range(starting_epoch, epochs):
        print(f'\n#-----------------------#\n     Epoch : {epoch + 1} / {epochs}\n#-----------------------#')
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].float().to(device)

            model.zero_grad()       
                
            result = model(batch_input_ids, 
                           attention_mask=batch_input_mask, 
                           labels=batch_labels,
                           return_dict=True)
            loss = result.loss
            logits = result.logits

            total_train_loss += loss.item()

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step == len(train_dataloader) - 1:
                avg_train_loss = total_train_loss / len(train_dataloader)

                total_eval_loss = 0
                total_eval_f1_micro = 0
                total_eval_accuracy = 0
                total_eval_f1_macro = 0
                model.eval()
                labels, logits = [], []

                for step, batch in enumerate(valid_dataloader):
                    batch_input_ids = batch[0].to(device)
                    batch_input_mask = batch[1].to(device)
                    batch_labels = batch[2].float().to(device)
                    with torch.no_grad():        
                        result = model(batch_input_ids, 
                               attention_mask=batch_input_mask,
                               labels=batch_labels,
                               return_dict=True)

                    loss = result.loss
                    logits = result.logits

                    total_eval_loss += loss.item()

                    logits = logits.detach().cpu().numpy()
                    label_ids = batch_labels.to('cpu').numpy()

                    total_eval_f1_micro += compute_f1_micro(get_probs(logits), label_ids)
                    total_eval_f1_macro += compute_f1_macro(get_probs(logits), label_ids)
                    total_eval_accuracy += flat_accuracy(get_probs(logits), label_ids)
                avg_val_accuracy = total_eval_accuracy / len(valid_dataloader)
                avg_val_f1_micro = total_eval_f1_micro / len(valid_dataloader)
                avg_val_f1_macro = total_eval_f1_macro / len(valid_dataloader)
                avg_val_loss = total_eval_loss / len(valid_dataloader)
        
                print(f'Average validation loss : {avg_val_loss:.3f}')
                print('Average validation metrics:')
                print('----------------')
                print(f'Accuracy : {avg_val_accuracy:.3f}')
                print(f'f1-score micro : {avg_val_f1_micro:.3f}')
                print(f'f1-score macro : {avg_val_f1_macro:.3f}')

                if highest_valid_acc == 0 or avg_val_accuracy > highest_valid_acc:
                    highest_valid_acc = avg_val_accuracy

                    patience = 0
                    # Save the best model:
                    PATH = output_dir
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.config.to_json_file(f"{PATH}/config.json")
                    unwrapped_model.save_pretrained(PATH, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(PATH)
                else:
                    patience += 1
                if early_stop:
                    if patience > 4:
                        break_flag1 = True
                        break
            
            if early_stop and (break_flag1 or completed_steps >= max_train_steps):
                break_flag2 = True
                break
        if early_stop and break_flag2:
            training_stats.append(
            {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'valid_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy,
                'val_f1_micro': avg_val_f1_micro,
                'val_f1_macro': avg_val_f1_macro,
                'early_stop': 1
            })
            print(f"Early stop with patience {patience}")
            break

        training_stats.append(
            {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'valid_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy,
                'val_f1_micro': avg_val_f1_micro,
                'val_f1_macro': avg_val_f1_macro,
                'early_stop': 0
            })
        
    output_file = output_stat_path
    print(training_stats)
    with open(output_file, 'w') as f:
        json.dump(training_stats, f, indent=4)
    print('Training finished...')

if __name__ == "__main__":
    train_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/all/train/train_dataset.parquet'
    val_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/all/val/val_dataset.parquet'
    test_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/all/test/test_dataset.parquet'
    # output_dir = 'D:/Elham/EnrichMyData/EnrichData_PC/model/distilbert_trained_model/999/label'
    output_dir = 'D:/Elham/EnrichMyData/EnrichData_PC/model/distilroberta_model/all/label'
    # output_stat_path = "D:/Elham/EnrichMyData/EnrichData_PC/model/distilbert_stats/999/training_stats.json"
    output_stat_path = "D:/Elham/EnrichMyData/EnrichData_PC/model/distilroberta_stats/all/training_stats.json"
    # model_name = 'distilbert-base-uncased'
    model_name = 'distilroberta-base'
    os.makedirs(output_dir, exist_ok=True)

    with open('D:/Elham/EnrichMyData/EnrichData_PC/data/splits/all/label_stats.json', 'r') as file:
            num_labels_data = json.load(file)
    n_labels = num_labels_data['num_labels']
    possible_labels = num_labels_data['possible_labels']
    main(train_file, val_file, test_file, output_dir, model_name, 'label', output_stat_path, n_labels, possible_labels)