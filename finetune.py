import os
import pdb
import json
import tqdm
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

import argparse
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomizedDataset(Dataset):
  def __init__(self, path, tokenizer, require_features=False):
    self.path = path
    self.tokenizer = tokenizer
    self.require_features = require_features
    with open(os.path.join(self.path, 'data.json')) as file:
      self.data = json.load(file)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    data = self.data[str(i)]

    p1_data = data['p1']
    p2_data = data['p2']
    label = data['label']

    if self.require_features:
      p1_features = data['p1_features']
      p2_features = data['p2_features']
      return {'p1_data':p1_data, 'p2_data':p2_data, 'p1_features':p1_features, 'p2_features':p2_features, 'label':label}
    else:
      return {'p1_data':p1_data, 'p2_data':p2_data, 'label':label}
    
def collate_fn(batch, tokenizer, require_features):
    p1_data = [item['p1_data'] for item in batch]
    p2_data = [item['p2_data'] for item in batch]
    labels = [item['label'] for item in batch]
    batch_len = len(labels)
    encoded_batch = tokenizer.batch_encode_plus(
        p1_data+p2_data,
        padding="max_length",
        truncation=True,
        max_length=256, 
        return_tensors="pt"
    )
    pdb.set_trace()
    p1_text, p2_text = encoded_batch["input_ids"][:batch_len,:], encoded_batch["input_ids"][batch_len:,:]
    if require_features:
        p1_features = torch.tensor([item['p1_features'] for item in batch])
        p2_features = torch.tensor([item['p2_features'] for item in batch])
        return p1_text, p2_text, F.normalize(p1_features,p=2,dim=1), F.normalize(p2_features,p=2,dim=1), torch.tensor(labels).float().to(device)
    else:
        return p1_text, p2_text, torch.tensor(labels).float().to(device)

class FTLogReg(nn.Module):
  def __init__(self, model_name, require_features):
    super().__init__()
    self.require_features = require_features
    self.pretrain = AutoModel.from_pretrained(model_name)
    self.dropout = nn.Dropout(0.1)
    input_size = self.pretrain.config.hidden_size
    if require_features:
      linear = nn.Linear((input_size+6)*2,512)
    else:
      linear = nn.Linear((input_size)*2,512)
    self.model = nn.Sequential(
                 linear,
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(512,1),
                 nn.Sigmoid()
                  )

  def forward(self, inputs):
    if self.require_features:
      t1, t2, f1, f2 = inputs
    else:
      t1, t2 = inputs
    e = self.pretrain(torch.cat((t1,t2),dim=0)).last_hidden_state[:,0,:]
    if self.require_features:
        input = torch.cat((torch.cat((e[:int(e.size(0)/2),:], f1),dim=1), torch.cat((e[int(e.size(0)/2):,:], f2),dim=1)), dim=1)
    else:
        input = torch.cat((e[:int(e.size(0)/2),:], e[int(e.size(0)/2):,:]), dim=1)
    return self.sigmoid(self.linear(input))

def evaluate(model, val_loader, criterion):
  model.eval()
  val_acc = .0
  val_f1 = .0
  val_loss = .0
  batch_len = len(val_loader)

  for batch in val_loader:
    inputs, labels = batch[:-1], batch[-1]
    outputs = model(inputs).reshape(-1)
    loss = criterion(outputs, labels)
    with torch.no_grad():
      val_loss += loss.item()
      val_acc += ((labels == (outputs > 0.5)).sum()/len(labels)).item()
      val_f1 += f1_score(labels.detach().cpu().numpy(), (outputs.detach().cpu().numpy() > 0.5))
  return val_loss/batch_len, val_acc/batch_len, val_f1/batch_len

def train():
  print_step = args['print_step']
  project_name = "LP2_Project_NF_finetune"
  best_model_path = 'best_model_finetune.pth'

  # Initilaize WanB
  wandb.init(project=project_name)
  config = wandb.config

  # set up config
  config.lr = args['learning_rate']
  config.batch_size = args['batch_size']
  config.num_epochs = args['n_epochs']
  config.optimizer = "adam"

  # set up training set and loader
  tokenizer = AutoTokenizer.from_pretrained(args['pretrained'])
  train_set = CustomizedDataset(path='train/', tokenizer=tokenizer,require_features=args['features'])
  val_set = CustomizedDataset(path="val/", tokenizer=tokenizer, require_features=args['features'])

  train_loader = DataLoader(train_set, batch_size=args['batch_size'],shuffle=True,collate_fn=lambda batch: collate_fn(batch, tokenizer, require_features=args['features']))
  val_loader = DataLoader(val_set, batch_size=args['batch_size'],collate_fn=lambda batch: collate_fn(batch, tokenizer, require_features=args['features'])) 

  # Model
  model = FTLogReg(model_name=args['pretrained'], require_features=args['features'])
  if (torch.cuda.device_count() > 1) and (device != torch.device("cpu")):
      model= nn.DataParallel(model)
  model.to(device)
  # Loss and Optimizer
  optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])  
  criterion = nn.BCELoss()
  # Scheduler
  def lr_lambda(step):
    if step < args['warm_up']:
        decay_factor = (step + 1) / args['warm_up']
        return decay_factor
    else:
        decay_factor = 0.95** ((step - args['warm_up']) // args['weight_decay'])
        return decay_factor
  scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
  # wandb.watch(model, criterion, log="all", log_freq = 100)

  best_f1 = 0
  # Main training Loop
  for epoch in range(args['n_epochs']):
    total_loss = 0 
    total_acc = 0
    total_f1 = 0
    ## for (in1, in2, labels) in tqdm(train_loader):
    for step, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
      # model feedforward
      if step == 150:
        torch.cuda.profiler.profile(enable=True)
        torch.cuda.profiler.start()
      model.train()
      inputs, labels = batch[:-1], batch[-1]
      outputs = model(inputs).reshape(-1)
      loss = criterion(outputs, labels.to(device))
      # Backward  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Train measurement 
      with torch.no_grad():
        total_loss += loss.item()
        total_acc += ((labels == (outputs > 0.5)).sum()/len(labels)).item()
        total_f1 += f1_score(labels.detach().cpu().numpy(), (outputs.detach().cpu().numpy() > 0.5))
      if step == 150:
        torch.cuda.profiler.stop()
        torch.cuda.profiler.profile(enable=False)
        print(torch.cuda.memory_allocated(device=torch.device("cuda:0")))
        print(torch.cuda.memory_allocated(device=torch.device("cuda:1")))
      # Print info
      if (step+1) % print_step == 0:
        with torch.no_grad():
          # Train set
          avg_loss, avg_acc, avg_f1 = total_loss/print_step, total_acc/print_step, total_f1/print_step
          # Val set
          val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
          # Save the best model
          if val_f1 > best_f1:
            best_f1 = avg_f1
            try:
              state_dict = model.module.state_dict()
            except AttributeError:
              state_dict = model.state_dict()
            # best_model_path = "f1" + "{:.3f}".format(avg_f1) + best_model_path
            torch.save(state_dict, best_model_path)
            print("Saved the best model!")
          print(f"Epoch [{epoch+1}/{args['n_epochs']}], Step [{step+1}], Train Avg. Loss: {avg_loss:.4f}, Train Avg. Acc: {avg_acc:.4f}, Train Avg. F1: {avg_f1:.4f}")
          print(f"Epoch [{epoch+1}/{args['n_epochs']}], Step [{step+1}], Val Avg. Loss: {val_loss:.4f}, Val Avg. Acc: {val_acc:.4f}, Val Avg. F1: {val_f1:.4f}")
          wandb.log( {"Epoch": epoch+1, "Step": step+1, "Train Avg. Loss": avg_loss, "Train Avg. Acc": avg_acc, "Train Avg. F1": avg_f1,  'Val Avg. Loss': val_loss, "Val Avg. Acc": val_acc, 'Val Avg. F1': val_f1})
          # Reset 
          total_loss = 0 
          total_acc = 0
          total_f1 = 0
      scheduler.step()
  print("Finished Training")

if __name__ == '__main__':
  # define ArgParser
  parser = argparse.ArgumentParser(description='Model Parser')
  parser.add_argument('-n','--n_epochs', default=5, type=int)
  parser.add_argument('-l','--learning_rate', default=1e-4, type=float)
  parser.add_argument('-b', '--batch_size', default=256, type=int)
  parser.add_argument('-print', '--print_step', default=50, type=int)
  parser.add_argument('-warm_up', '--warm_up', default=20, type=int)
  parser.add_argument('-weight_decay', '--weight_decay', default=25, type=int)
  parser.add_argument('-features', '--features', action='store_true', default=False)
  parser.add_argument('-p','--pretrained', default='bert-base-uncased', type=str)
  args = parser.parse_args().__dict__
  
  train()
