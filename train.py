import os
import json
import tqdm
import glob
import nltk
# import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomizedDataset(Dataset):
  def __init__(self, path, require_features=False):
    self.path = path
    self.require_features = require_features
    with open(os.path.join(self.path, 'data.json')) as file:
      self.data = json.load(file)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    data = self.data[str(i)]

    p1_embedding = torch.load(os.path.join(self.path, 'embeddings', 'data_'+str(i+1)+'_p1.pt'), map_location=torch.device(device))
    p2_embedding = torch.load(os.path.join(self.path, 'embeddings', 'data_'+str(i+1)+'_p2.pt'), map_location=torch.device(device))
    label = data['label']

    if self.require_features:
      p1_features = data['p1_features']
      p2_features = data['p2_features']
      return p1_embedding, p2_embedding, torch.tensor(p1_features), torch.tensor(p2_features), torch.tensor(label)
    else:
      return p1_embedding, p2_embedding, torch.tensor(label)

class LogReg(nn.Module):
  def __init__(self,input_ln):
    super().__init__()
    self.model = nn.Sequential(
                 nn.Linear(input_ln,1),
                 nn.Sigmoid()
                  )

  def forward(self, input):
    return self.model(input)

def train():

  project_name = "LP2_Project_NF"
  best_model_path = '.pth'

  # Initilaize WanB
  # wandb.init(project=project_name)
  # config = wandb.config

  # set up config
  # config.lr = 0.001
  # config.batch_size = 16
  # config.num_epochs = 1
  # config.optimizer = "sgd"

  print("Loading data:")
  # set up training set and loader
  train_set = CustomizedDataset('/content/drive/MyDrive/LP2_ProjectData/release/pan23-multi-author-analysis-dataset2/training/')
  print("Train set loaded")
  train_loader = DataLoader(train_set, batch_size=16,shuffle=True) 
  print("Train dataloader created")

  for i in train_set:
    input_len = i[0].shape[0] # get size of combined vector from dataloader / dataset
    break
  model = LogReg(input_len*2)
  model.to(device)
  # if config.optimizer == "sgd":
  #   optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
  # elif config.optimizer == "adam":
  #   optimizer = optim.Adam(model.parameters(), lr=config.lr)
  optimizer = optim.SGD(model.parameters(), lr=0.00001)  
  criterion = nn.BCELoss()
  # wandb.watch(model, criterion, log="all", log_freq = 100)

  best_f1 = 0
  model.train()
  # Main training Loop
  for epoch in range(10):
    running_f1 = 0
    running_loss = 0 

    step = 0
    ## for (in1, in2, labels) in tqdm(train_loader):
    for (in1, in2, labels) in train_loader:
      # model feedforward
      inputs = torch.hstack([in1,in2])
      outputs = model(inputs).reshape(-1)
      labels = labels.type(torch.float)
      loss = criterion(outputs, labels)

      # backprop  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      if (step+1) % 10 == 0:
        with torch.no_grad():
          avg_f1 = running_f1 / 100
          avg_loss = running_loss / 100

          print(f"Epoch [{epoch+1}/{15}], Step [{step+1}], Avg. Loss: {avg_loss:.4f}, Avg. F1: {avg_f1:.4f}" )
          # wandb.log( {"Epoch": epoch+1, "Step": step+1, "Avg. Loss": avg_loss, "Avg. F1": avg_f1 })
      step +=1
  # Save the best model
  if avg_f1 > best_f1:
    best_f1 = avg_f1
    try:
      state_dict = model.module.state_dict()
    except AttributeError:
      state_dict = model.state_dict()
    best_model_path = "f1" + "{:.5f}".format(avg_f1) + best_model_path
    torch.save(state_dict, best_model_path)
    print("Saved the best model!")

  print("Finished Training")


if '__name__' == '__main__':
  train()

