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
  def __init__(self, path, tokenizer, max_len, require_features=False):
    self.path = path
    self.tokenizer = tokenizer
    self.max_len = max_len
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
    p1 = self.tokenizer.encode_plus(
            p1_data,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
    p1_ids = p1['input_ids'].squeeze(0)
    p1_mask = p1['attention_mask'].squeeze(0)
    p2 = self.tokenizer.encode_plus(
            p2_data,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
    p2_ids = p2['input_ids'].squeeze(0)
    p2_mask = p2['attention_mask'].squeeze(0)
    if self.require_features:
      p1_features = torch.tensor(data['p1_features'])
      p2_features = torch.tensor(data['p2_features'])
      norm = torch.norm(p1_features, p=2)
      p1_features = p1_features / norm 
      norm = torch.norm(p2_features, p=2)
      p2_features = p2_features / norm 
      return {'p1_ids':p1_ids, 'p1_mask':p1_mask, 'p2_ids':p2_ids, 'p2_mask':p2_mask, 'p1_features':p1_features, 'p2_features':p2_features, 'label':label}
    else:
      return {'p1_ids':p1_ids, 'p1_mask':p1_mask, 'p2_ids':p2_ids, 'p2_mask':p2_mask, 'label':label}

class FTLogReg(nn.Module):
  def __init__(self, model_name, require_features, require_finetune):
    super().__init__()
    self.require_features = require_features
    self.pretrain = AutoModel.from_pretrained(model_name)
    if not require_finetune:
      for param in self.pretrain.parameters():
        param.requires_grad = False
    self.dropout = nn.Dropout(0.3)
    input_size = self.pretrain.config.hidden_size
    if require_features:
      linear = nn.Linear((input_size+6)*2,512)
    else:
      linear = nn.Linear((input_size)*2,input_size)
    self.model = nn.Sequential(
                 linear,
                 nn.ReLU(),
                 nn.Dropout(0.3),
                 nn.Linear(input_size,1),
                 nn.Sigmoid()
                  )

  def forward(self, ids, masks, features=None):
    # Load Data
    t1, t2 = ids
    m1, m2 = masks
    if self.require_features:
      f1, f2 = features
    # Forward Pretrain
    e = self.pretrain(torch.cat((t1,t2),dim=0),torch.cat((m1,m2),dim=0)).last_hidden_state[:,0,:]
    e = self.dropout(e)
    # Forward Linear
    if self.require_features:
        input = torch.cat((torch.cat((e[:int(e.size(0)/2),:], f1),dim=1), torch.cat((e[int(e.size(0)/2):,:], f2),dim=1)), dim=1)
    else:
        input = torch.cat((e[:int(e.size(0)/2),:], e[int(e.size(0)/2):,:]), dim=1)
    return self.model(input)

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
  project_name = "LP2_Project"
  
  # Set save path
  best_model_path = args['model'] + '_' +args['train'][:2]
  if args['features']:
    best_model_path += '_features'
  if args['finetune']:
    best_model_path += '_finetune'
  best_model_path += '.pth'

  # Initilaize WanB
  wandb.init(project=project_name)
  config = wandb.config

  # set up config
  config.lr = args['learning_rate']
  config.batch_size = args['batch_size']
  config.num_epochs = args['n_epochs']
  config.optimizer = "adamw"

  # set up training set and loader
  tokenizer = AutoTokenizer.from_pretrained(args['model'])
  train_set = CustomizedDataset(path=args['train'], tokenizer=tokenizer, max_len=args['max_len'], require_features=args['features'])
  val_set = CustomizedDataset(path=args['val'], tokenizer=tokenizer, max_len=args['max_len'], require_features=args['features'])

  train_loader = DataLoader(train_set, batch_size=args['batch_size'],shuffle=True)
  val_loader = DataLoader(val_set, batch_size=args['batch_size']) 

  # Model
  model = FTLogReg(model_name=args['model'], require_features=args['features'], require_finetune=args['finetune'])
  # if (torch.cuda.device_count() > 1) and (device != torch.device("cpu")):
  #     model= nn.DataParallel(model)
  model.to(device)
  # Loss and Optimizer
  optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])  
  criterion = nn.BCELoss()
  # Scheduler
  if args['finetune']:
    def lr_lambda(step):
      if step < args['warm_up']:
          decay_factor = (step + 1) / args['warm_up']
          return decay_factor
      else:
          decay_factor = 0.95** ((step - args['warm_up']) // args['weight_decay'])
          return decay_factor
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

  best_f1 = 0
  # Main training Loop
  for epoch in range(args['n_epochs']):
    total_loss = 0 
    total_acc = 0
    total_f1 = 0
    ## for (in1, in2, labels) in tqdm(train_loader):
    for step, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
      # model feedforward
      model.train()
      pdb.set_trace()
      ids, masks, labels = (batch['p1_ids'].to(device), batch['p2_ids'].to(device)), (batch['p1_mask'].to(device), batch['p2_mask'].to(device)), batch['labels']
      features = None
      if args['features']:
        features = (batch['p1_features'].to(device), batch['p2_features'].to(device))
      outputs = model(ids, masks, features).reshape(-1)
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
      if args['finetune']:
        scheduler.step()
  print("Finished Training")

if __name__ == '__main__':
  # define ArgParser
  parser = argparse.ArgumentParser(description='Model Parser')
  parser.add_argument('-n','--n_epochs', default=5, type=int)
  parser.add_argument('-l','--learning_rate', default=1e-5, type=float)
  parser.add_argument('-b', '--batch_size', default=32, type=int)
  parser.add_argument('-max_len', '--max_len', default=256, type=int)
  parser.add_argument('-print', '--print_step', default=100, type=int)
  parser.add_argument('-warm_up', '--warm_up', default=200, type=int)
  parser.add_argument('-weight_decay', '--weight_decay', default=250, type=int)
  parser.add_argument('-features', '--features', action='store_true', default=False)
  parser.add_argument('-finetune', '--finetune', action='store_true', default=False)
  parser.add_argument('-model','--model', default='bert-base-uncased', type=str)
  parser.add_argument('-train','--train', default='d2_train', type=str)
  parser.add_argument('-val','--val', default='d2_val', type=str)

  args = parser.parse_args().__dict__
  
  train()
