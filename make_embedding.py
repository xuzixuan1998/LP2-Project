import os
import pdb
import json
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_path = 'val/data.json'
output_path = 'val/embeddings'
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

with open(json_path) as f:
    datas = json.load(f)
for i, data in datas.items():
    # Bert Embedding
    encoded = tokenizer.encode_plus(data['p1'],
                    padding="longest",
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt")
    output = model(encoded["input_ids"].to(device), encoded['attention_mask'].to(device))
    torch.save(output.last_hidden_state[0][0], os.path.join(output_path,'data_'+i+'_p1.pt')) 
    encoded = tokenizer.encode_plus(data['p2'],
                    padding="longest",
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt")
    output = model(encoded["input_ids"].to(device), encoded['attention_mask'].to(device))
    torch.save(output.last_hidden_state[0][0], os.path.join(output_path,'data_'+i+'_p2.pt'))   

print("Done")