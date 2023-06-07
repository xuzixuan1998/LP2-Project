import os
import pdb
import json
import tqdm
import glob
import nltk
# import wandb
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class CustomizedDataset(Dataset):
#   def __init__(self, path, require_features=False):
#     self.path = path
#     self.require_features = require_features
#     with open(os.path.join(self.path, 'data.json')) as file:
#       self.data = json.load(file)

#   def __len__(self):
#     return len(self.data)

#   def __getitem__(self, i):
#     pdb.set_trace()
#     data = self.data[str(i)]

#     p1_embedding = torch.load(os.path.join(self.path, 'embeddings', 'data_'+str(i)+'_p1.pt'), map_location=torch.device(device))
#     p2_embedding = torch.load(os.path.join(self.path, 'embeddings', 'data_'+str(i)+'_p2.pt'), map_location=torch.device(device))
#     label = data['label']

#     if self.require_features:
#       p1_features = data['p1_features']
#       p2_features = data['p2_features']
#       return torch.cat((p1_embedding, F.normalize(torch.tensor(p1_features).to(device),p=2))), torch.cat((p2_embedding, F.normalize(torch.tensor(p2_features).to(device),p=2))), label
#     else:
#       return p1_embedding, p2_embedding, label

class FTLogReg(nn.Module):
  def __init__(self,input_ln, model_name):
    super().__init__()
    self.pretrain = AutoModel.from_pretrained(model_name)
    self.model = nn.Sequential(
                 nn.Linear(input_ln,1),
                 nn.Sigmoid()
                  )

  def forward(self, inputs):
    t1, t2, f1, f2 = inputs
    e1 = self.pretrain(t1).last_hidden_state[0][0]
    e2 = self.pretrain(t2).last_hidden_state[0][0]
    input = torch.cat((torch.cat((e1, f1),dim=1), torch.cat((e2, f2),dim=1)), dim=1)
    return self.model(input)

def evaluate(model, val_loader, criterion):
  model.eval()
  val_f1 = .0
  val_loss = .0
  batch_len = len(val_loader)

  for (in1, in2, labels) in val_loader:
    inputs = torch.cat((in1,in2), dim=1)
    outputs = model(inputs).reshape(-1)
    loss = criterion(outputs, labels)
    val_loss += loss.item()
    val_f1 += f1_score(labels, (outputs > 0.5).detach().cpu().numpy())
  return val_loss/batch_len, val_f1/batch_len



def train():

  print_step = 100
  project_name = "LP2_Project_NF"
  best_model_path = 'best_model.pth'

  # Initilaize WanB
  # wandb.init(project=project_name)
  # config = wandb.config

  # set up config
  # config.lr = 0.001
  # config.batch_size = 16
  # config.num_epochs = 1
  # config.optimizer = "sgd"

  # set up training set and loader
  train_set = CustomizedDataset('train/', require_features=args['features'])
  val_set = CustomizedDataset(data_path="val/", require_features=args['features'])

  train_loader = DataLoader(train_set, batch_size=args['batch_size'],shuffle=True)
  val_loader = DataLoader(val_set, batch_size=args['batch_size']) 

  # Model
  if args['features']:
    model = LogReg(768*2)
  else:
    model = LogReg(input_len*2)
  if (torch.cuda.device_count() > 1) and (device != torch.device("cpu")):
      model= nn.DataParallel(model)
  model.to(device)
  # Loss and Optimizer
  optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'])  
  criterion = nn.BCELoss()
  # wandb.watch(model, criterion, log="all", log_freq = 100)

  best_f1 = 0
  # Main training Loop
  for epoch in range(args['n_epochs']):
    total_loss = 0 
    total_f1 = 0
    step = 0
    ## for (in1, in2, labels) in tqdm(train_loader):
    for (in1, in2, labels) in train_loader:
      # model feedforward
      model.train()
      inputs = torch.cat((in1,in2), dim=1)
      outputs = model(inputs).reshape(-1)
      loss = criterion(outputs, labels)
      # Backward  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Train measurement 
      total_loss += loss.item()
      total_f1 += f1_score(labels, (outputs > 0.5).detach().cpu().numpy())
      # Print info
      if (step+1) % print_step == 0:
        with torch.no_grad():
          # Train set
          avg_loss, avg_f1 = total_loss/print_step, total_f1/print_step
          # Val set
          val_loss, val_f1 = evaluate(model, val_loader, criterion)
          # Save the best model
          if val_f1 > best_f1:
            best_f1 = avg_f1
            try:
              state_dict = model.module.state_dict()
            except AttributeError:
              state_dict = model.state_dict()
            best_model_path = "f1" + "{:.3f}".format(avg_f1) + best_model_path
            torch.save(state_dict, best_model_path)
            print("Saved the best model!")
          # Logging
          print(f"Epoch [{epoch+1}/{args['n_epochs']}], Step [{step+1}], Train Avg. Loss: {avg_loss:.4f}, Train Avg. F1: {avg_f1:.4f}")
          print(f"Epoch [{epoch+1}/{args['n_epochs']}], Step [{step+1}], Val Avg. Loss: {val_loss:.4f}, Val Avg. F1: {val_f1:.4f}")
          # wandb.log( {"Epoch": epoch+1, "Step": step+1, "Avg. Loss": avg_loss, "Avg. F1": avg_f1 })
      step +=1
  print("Finished Training")

if __name__ == '__main__':
  # define ArgParser
  parser = argparse.ArgumentParser(description='Model Parser')
  parser.add_argument('-e','--n_epochs', default=50, type=int)
  parser.add_argument('-l','--learning_rate', default=1e-4, type=float)
  parser.add_argument('-b', '--batch_size', default=256, type=int)
  parser.add_argument('-warm_up', '--warm_up', default=20, type=int)
  parser.add_argument('-weight_decay', '--weight_decay', default=25, type=int)
  parser.add_argument('-features', '--features', action='store_true', default=False)
  parser.add_argument('-p','--pretrained', default='bert-base-uncased', type=str)
  args = parser.parse_args().__dict__

  train()