import os
import pdb
import json
import tqdm
import wandb
import argparse
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

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
    self.model_name = model_name
    self.pretrain = AutoModel.from_pretrained(model_name)
    if not require_finetune:
      for param in self.pretrain.parameters():
        param.requires_grad = False
    self.dropout = nn.Dropout(0.3)
    input_size = self.pretrain.config.hidden_size
    if require_features:
      linear = nn.Linear((input_size+6)*2,input_size)
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
    if self.model_name == 'bert-base-uncased':
      pooler_output = self.pretrain(torch.cat((t1,t2),dim=0),torch.cat((m1,m2),dim=0)).pooler_output
    else:
      pooler_output = self.pretrain(torch.cat((t1,t2),dim=0),torch.cat((m1,m2),dim=0)).last_hidden_state[:,0,:]
    e = self.dropout(pooler_output)
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
    ids, masks, labels = (batch['p1_ids'].to(device), batch['p2_ids'].to(device)), (batch['p1_mask'].to(device), batch['p2_mask'].to(device)), batch['label'].float().to(device)
    features = None
    if args['features']:
      features = (batch['p1_features'].to(device), batch['p2_features'].to(device))
    outputs = model(ids, masks, features).reshape(-1)
    loss = criterion(outputs, labels)
    with torch.no_grad():
      val_loss += loss.item()
      val_acc += ((labels == (outputs > 0.5)).sum()/len(labels)).item()
      val_f1 += f1_score(labels.detach().cpu().numpy(), (outputs.detach().cpu().numpy() > 0.5))
  return val_loss/batch_len, val_acc/batch_len, val_f1/batch_len

def generate_saliency_map(model, val_loader):
    data = {}
    for i, batch in enumerate(val_loader):
      # if len(data) == 10:
      #   break
      tokens, ids, masks, labels = (batch['p1_data'], batch['p2_data']), (batch['p1_ids'].to(device), batch['p2_ids'].to(device)), (batch['p1_mask'].to(device), batch['p2_mask'].to(device)), batch['label'].float().to(device)
      if (len(tokens[0]) > 60) or (len(tokens[1]) > 60):
        continue
      features = None
      if args['features']:
        features = (batch['p1_features'].to(device), batch['p2_features'].to(device))
      # Convert input tokens to tensor
      features = (features[0].requires_grad_(), features[1].requires_grad_())
      # Forward pass to get model predictions
      model.pretrain.embeddings.word_embeddings.weight.requires_grad_()
      model.zero_grad()
      outputs = model(ids, masks, features).reshape(-1)
      if ((outputs > 0.5) == labels) and ((outputs.item() > 0.95) or (outputs.item() < 0.05)):
        # Calculate gradients
        outputs.sum().backward()
        # Get the gradients of the input tensor
        embedding_gradients = model.pretrain.embeddings.word_embeddings.weight.grad
        idx1, idx2 = ids[0][ids[0] != 0], ids[1][ids[1] != 0]
        token_gradient_1, token_gradient_2 = torch.abs(embedding_gradients[idx1]).mean(), torch.abs(embedding_gradients[idx1]).mean()
        ids_gradients_1, ids_gradients_2 =torch.norm(embedding_gradients[idx1], p=2, dim=1), torch.norm(embedding_gradients[idx2], p=2, dim=1)
        features_gradients_1, features_gradients_2 = torch.abs(features[0].grad[0]), torch.abs(features[1].grad[0])
        # Normalize gradients
        ids_gradients_1 /= ids_gradients_1.max()
        ids_gradients_2 /= ids_gradients_2.max()
        features_gradients_1 = torch.hstack([token_gradient_1, features_gradients_1])
        features_gradients_1 /= features_gradients_1.max()
        features_gradients_2 = torch.hstack([token_gradient_2, features_gradients_2])
        features_gradients_2 /= features_gradients_2.max()
        data[len(data)] = {'p1':tokens[0], 'p2':tokens[1], 'p1_gradients':ids_gradients_1.detach().cpu().numpy().tolist(), 'p2_gradients':ids_gradients_2.detach().cpu().numpy().tolist(), 'feature1_gradients':features_gradients_1.detach().cpu().numpy().tolist(), 'feature2_gradients':features_gradients_2.detach().cpu().numpy().tolist(), 'lebel':labels.item()}
    with open('saliency.json', 'w') as f:
      json.dump(data, f)

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
  val_loader = DataLoader(val_set, batch_size=args['batch_size'],shuffle=True) 

  # Model
  model = FTLogReg(model_name=args['model'], require_features=args['features'], require_finetune=args['finetune'])
  # Loss and Optimizer
  optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])  
  criterion = nn.BCELoss()
      # If only evaluation
  if args['test'] or args['saliency']:
    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    if args['saliency']:
      val_loader = DataLoader(val_set, batch_size=1,shuffle=True) 
      generate_saliency_map(model, val_loader)
    if args['test']:
      val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
      print(f"Model: {best_model_path}, Val Avg. Loss: {val_loss:.4f}, Val Avg. Acc: {val_acc:.4f}, Val Avg. F1: {val_f1:.4f}")
    return
  # Multi-GPU
  if (torch.cuda.device_count() > 1) and (device != torch.device("cpu")):
      model= nn.DataParallel(model)
  model.to(device)
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
      ids, masks, labels = (batch['p1_ids'].to(device), batch['p2_ids'].to(device)), (batch['p1_mask'].to(device), batch['p2_mask'].to(device)), batch['label'].float().to(device)
      features = None
      if args['features']:
        features = (batch['p1_features'].to(device), batch['p2_features'].to(device))
      outputs = model(ids, masks, features).reshape(-1)
      loss = criterion(outputs, labels)
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
  parser.add_argument('-test', '--test', action='store_true', default=False)
  parser.add_argument('-saliency', '--saliency', action='store_true', default=False)
  parser.add_argument('-model','--model', default='bert-base-uncased', type=str)
  parser.add_argument('-train','--train', default='d2_train', type=str)
  parser.add_argument('-val','--val', default='d2_val', type=str)

  args = parser.parse_args().__dict__
  
  train()
