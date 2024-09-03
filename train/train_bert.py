import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertModel,  get_scheduler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
import pandas as pd
from datasets import load_dataset, load_metric, concatenate_datasets
import torch.nn as nn
import datasets
import transformers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BERTClass(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output
    
def compute_f1_micro(out, pred):
    return metrics.f1_score(pred, out, average='micro', zero_division=1)

def compute_f1_macro(out, pred):
    return metrics.f1_score(pred, out, average='macro', zero_division=1)

def flat_accuracy(preds, labels):
    res = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        res[i] = np.all(preds[i] == labels[i])
    return np.sum(res) / labels.shape[0]

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
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'labels': labels, 'text': text}
    


def main(train_file, val_file, test_file, output_dir, model_name, label, output_stat_path):
    seed = 42
    batch_size = 32
    learning_rate = 4e-5
    epochs = 5

    lr_scheduler_type = "linear" #choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    num_warmup_steps = 0
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    early_stop = True

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

    # Load data
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_df[label].tolist())
    val_labels = mlb.transform(val_df[label].tolist())
    test_labels = mlb.transform(test_df[label].tolist())

    train_dataset = CustomDataset(train_df['text'].tolist(), train_labels.tolist(), tokenizer)
    val_dataset = CustomDataset(val_df['text'].tolist(), val_labels.tolist(), tokenizer)
    test_dataset = CustomDataset(test_df['text'].tolist(), test_labels.tolist(), tokenizer)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    valid_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size)



    # Model
    num_labels = len(mlb.classes_)
    model =  BERTClass(num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
    

    # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. 
    # This version is more numerically stable than using a plain Sigmoid followed 
    # by a BCELoss as, by combining the operations into one layer, 
    # we take advantage of the log-sum-exp trick for numerical stability.
    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    def get_probs(logits, threshold=0.5):
        return torch.sigmoid(logits) >= threshold
    
    starting_epoch = 0
    completed_steps = 0
    break_flag1, break_flag2 = False, False
    training_stats = []
    lowest_valid_loss, highest_valid_acc, patience = 0, 0, 0

    for epoch in range(starting_epoch, epochs):
        print(f'\n#-----------------------#\n     Epoch : {epoch + 1} / {epochs}\n#-----------------------#')
        model.train()
        total_train_loss = 0
        losses = []
        correct_predictions = 0
        num_samples = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device, dtype = torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.float)
            
             # forward
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            total_train_loss += loss.item()
            

           # training accuracy, apply sigmoid, round (apply thresh 0.5)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            labels = labels.cpu().detach().numpy()
            correct_predictions += np.sum(outputs==labels)
            num_samples += labels.size   # total number of elements in the 2D array


            # backward
            optimizer.zero_grad()
            accelerator.backward(loss)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    input_ids = batch['input_ids'].to(device, dtype = torch.long)
                    attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
                    token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
                    labels = batch['labels'].to(device, dtype = torch.float)
                
                    # forward
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    logits = outputs

                    loss = loss_fn(logits, labels)
                    losses.append(loss.item())
                    total_eval_loss += loss.item()

                    outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
                    logits = outputs
                    targets = labels.cpu().detach().numpy()
                    correct_predictions += np.sum(outputs==targets)
                    num_samples += targets.size   # total number of elements

                    total_eval_f1_micro += compute_f1_micro(logits, targets)
                    total_eval_f1_macro += compute_f1_macro(logits, targets)
                    total_eval_accuracy += flat_accuracy(logits, targets)

                avg_val_accuracy = total_eval_accuracy / len(valid_dataloader)
                avg_val_loss = total_eval_loss / len(valid_dataloader)
                avg_val_f1_micro = total_eval_f1_micro / len(valid_dataloader)
                avg_val_f1_macro = total_eval_f1_macro / len(valid_dataloader)

                print(f'Validation Loss: {avg_val_loss:.3f}')
                print(f'F1-score Micro: {avg_val_f1_micro:.3f}')
                print(f'F1-score Macro: {avg_val_f1_macro:.3f}')
                print(f'Accuracy : {avg_val_accuracy:.3f}')

                if highest_valid_acc == 0 or avg_val_accuracy > highest_valid_acc:
                    highest_valid_acc = avg_val_accuracy

                    patience = 0
                    # Save the best model:
                    PATH = output_dir
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    # unwrapped_model.config.to_json_file(f"{PATH}/config.json")
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
    train_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/99/train/train_dataset.parquet'
    val_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/99/val/val_dataset.parquet'
    test_file = 'D:/Elham/EnrichMyData/EnrichData_PC/data/splits/99/test/test_dataset.parquet'
    output_dir = 'D:/Elham/EnrichMyData/EnrichData_PC/model/bert_model/99/label'
    output_stat_path = "D:/Elham/EnrichMyData/EnrichData_PC/model/bert_model/99/training_stats.json"
    model_name = 'bert-base-uncased'

    main(train_file, val_file, test_file, output_dir, model_name, 'label', output_stat_path)
