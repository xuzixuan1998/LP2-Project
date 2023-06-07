import os
import json
import torch
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_path = 'train/data.json'
output_path = 'train/embeddings'
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

with open(json_path) as f:
    datas = json.load(f)
for i, data in datas.items():
    # Bert Embedding
    encoded = tokenizer.encode_plus(data['p1'],
                    padding="longest",
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt")
    output = model(encoded["input_ids"].to(device))
    torch.save(output.last_hidden_state[0][0], os.path.join(output_path,'data_'+str(i+1)+'_p1.pt')) 
    encoded = tokenizer.encode_plus(data['p2'],
                    padding="longest",
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt")
    output = model(encoded["input_ids"].to(device))
    torch.save(output.last_hidden_state[0][0], os.path.join(output_path,'data_'+str(i+1)+'_p2.pt'))   

print("Done")