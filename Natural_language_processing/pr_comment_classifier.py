#Importing libraries
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification
import pandas as pd
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
import torch.nn as nn
import wandb
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.optim import Adam
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder

import re
import string
import ast
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

# import pytorch_lightning as pl
# from torchmetrics.functional import accuracy, f1_score, auroc
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm.auto import tqdm

#Accessing Wandb
os.environ['WANDB_API_KEY']='b3d44a1ccb9147d1d5c05f7769f5d4521796380e'
os.environ['WANDB_ENTITY']='amirrshams'

wandb.login()


run = wandb.init(
    # Set the project where this run will be logged
    project="pr_classification",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 1e-8,
        "epochs": 200,
        "batch size": 24,
        "Dropout": 0.2,
        "train size":0.8,}
    )


#reading the data
df = pd.read_csv('/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/dataset/Sample_5000_manual.csv')

df = df.drop(['api_url', ' url', 'pr_url', 'pr_api_url', 'author_id', 'author_desc_body', 'closer_id','commit_counts', 'code_changes_counts', 'created_at', 'closed_at', 'author_country', 'author_continent', 'same_country', 'author_eth', 'closer_eth','closer_country', 'same_eth', 'prs_white', 'prs_black', 'prs_api', 'prs_hispanic', 'pri_white', 'pri_black', 'pri_api', 'pri_hispanic', 'prs_eth_7', 'prs_eth_8', 'prs_eth_9', 'prs_eth_diff', 'prs_eth_diff_2'], axis=1)

#text preprocessing
def text_preprocess(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'@[A-Za-z0-9]+','',text) #remove @mentions
    text = re.sub(r'#','',text) #remove # symbol
    text = re.sub(r'https?:\/\/\S+','',text) #remove the hyper link
    text = re.sub(r'\n','',text) #remove \n
    text = re.sub(r'www\S+', '', text) #remove www
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)     # Handle special characters and symbols
    text = re.sub(r'\b([a-f0-9]{40})\b', 'Commit ID', text)
    text = text.translate(str.maketrans('', '', string.punctuation))     # Remove punctuation

    return text
df['comments'] = df['comments'].apply(lambda x: ast.literal_eval(x))
df['comments'] = df['comments'].apply(lambda x: [text_preprocess(t) for t in x])
df['comments'] = df['comments'].apply(lambda x: ' '.join(x))
df['comments'] = df['comments'].apply(lambda x: x if len(x) > 0 else 'No comments')

#model

# creating the pytorch dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

labels = {'No reason':0, 'Unnecessary':1, 'Replaced': 2, 'Merge Conflict':3,
      'Successful':4, 'Stale':5, 'Resolved':6, 'Quality':7, 'Duplicate':8,
      'Chaotic':9, 'Not PR':10}

class PRDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['manual_analysis']]
        self.comment = [tokenizer(comment,
                                  padding = 'max_length',
                                  max_length = 512,
                                  truncation = True,
                                  return_tensors = 'pt') for comment in df['comments']]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_comments(self, idx):
        return self.comment[idx]
    
    def __getitem__(self, idx):

        batch_comments = self.get_batch_comments(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_comments, batch_y
        


#splitting the data
np.random.seed(112)

df_train, df_remaining = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_remaining, test_size=0.5, random_state=42)

print(len(df_train),len(df_val), len(df_test))

#bert classifier class
class BertClassifier(nn.Module):
    

    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout1 = nn.Dropout(0.2)
        #self.dropout2 = nn.Dropout(0.1) # added another dropout layer
        self.linear = nn.Linear(768, 11)
        # self.linear2 = nn.Linear(11, 11)
        self.softmax = nn.Softmax(dim=1) # changed relu to softmax
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        pooled_output = self.dropout1(pooled_output)
        pooled_output = self.linear(pooled_output)
        # pooled_output = self.linear2(pooled_output)
        #pooled_output = self.dropout2(pooled_output) # added another dropout layer
        final_layer = self.softmax(pooled_output) # changed relu to softmax
        return final_layer
  
#training the model

def train_modified(model, train_data, val_data, learning_rate, epochs):
    train_dataset, val_dataset = PRDataset(train_data), PRDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size = 24, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size=24)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = learning_rate)


    model = model.to(device)
    criterion = criterion.to(device)
    
    # if use_cuda:
    #     model = model.cuda()
    #     criterion = criterion.cuda()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch_num in tqdm(range(epochs)):
        
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in train_dataloader:

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # pooled_output = output.pooler_output
            # pooled_output = output[:, 0]

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # scheduler.step(batch_loss)
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device).float()
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                # pooled_output = output[:, 0]
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                # scheduler.step(batch_loss)
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        
        train_loss = total_loss_train / len(train_data)
        val_loss = total_loss_val / len(val_data)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracy = total_acc_train / len(train_data)
        val_accuracy = total_acc_val / len(val_data)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        # print(
        #     f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
        #     | Train Accuracy: {total_acc_train / len(train_data): .3f} \
        #     | Val Loss: {total_loss_val / len(val_data): .3f} \
        #     | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {train_loss:.3f} \
            | Train Accuracy: {train_accuracy:.3f} \
            | Val Loss: {val_loss:.3f} \
            | Val Accuracy: {val_accuracy:.3f}')
        
        wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy ,"Val Loss":val_loss, "Val Accuracy":val_accuracy})

        #plotting

    # epochs_list = range(1, epochs + 1)
    # plt.plot(epochs_list, train_losses, label='Train Loss')
    # plt.plot(epochs_list, val_losses, label='Validation Loss')
    # plt.plot(epochs_list, train_accuracies, label = 'Train Accuracy')
    # plt.plot(epochs_list, val_accuracies, label = 'Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
              
    
def evaluate(model, test_data):

    test = PRDataset(test_data)

    test_dataloader = DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)

    y_true = []
    y_pred = []
    total_acc_test = 0

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            # Compute predictions
            batch_pred = output.argmax(dim=1).cpu().numpy()
            batch_true = test_label.cpu().numpy()

            y_pred.extend(batch_pred.tolist())
            y_true.extend(batch_true.tolist())

            # Compute accuracy
            acc = (batch_pred == batch_true).sum().item()
            total_acc_test += acc

    # Compute metrics
    accuracy = total_acc_test / len(test_data)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f'Test Accuracy: {accuracy:.3f}')
    print(f"Classification Report: {report}")



model = BertClassifier()
train_modified(model, df_train, df_val, 1e-8, 200)

evaluate(model, df_test)

