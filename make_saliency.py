import pdb
import torch
from transformers import AutoTokenizer
from finetune import FTLogReg

def generate_saliency_map(model, i1, i2):
    # Set model in evaluation mode
    model.eval()

    # Convert input tokens to tensor
    ids = (i1['input_ids'].float().requires_grad_(), i2['input_ids'].float().requires_grad_())
    masks = (i1['attention_mask'], i2['attention_mask'])
    features = (torch.tensor(i1['input_features']).float().unsqueeze(0).requires_grad_(), torch.tensor(i2['input_features']).float().unsqueeze(0).requires_grad_())
    # Forward pass to get model predictions
    output = model(ids, masks, features)

    # Calculate gradients
    model.zero_grad()
    output.sum().backward()  # Backward pass

    # Get the gradients of the input tensor
    ids_gradients = torch.tensor([torch.abs(ids[0].grad[0]), torch.abs(ids[1].grad[0])])
    features_gradients = torch.tensor([torch.abs(features[0].grad[0]), torch.abs(features[1].grad[0])])
    # Normalize gradients
    ids_gradients = torch.abs(ids_gradients)
    ids_gradients /= ids_gradients.max(dim=1)
    features_gradients = torch.abs(features_gradients)
    features_gradients /= features_gradients.max(dim=1)


    return ids_gradients.detach().numpy(), features_gradients.detach().numpy()

model_name = 'bert-base-uncased'

model = FTLogReg(model_name, True, True)
state_dict = torch.load('bert-base-uncased_d2_features_finetune.pth')
model.load_state_dict(state_dict)
t1 = "They're responsible for the upkeep of the garage. They're not responsible for the actions of other individuals within the garage."
t2 = "But threatening a lawyer won't amount to much, they weren't the ones who stole from you. You would need to sue that person if caught for the value of the items."
tokenizer = AutoTokenizer.from_pretrained(model_name)
i1 = tokenizer.encode_plus(
            t1,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
i2 = tokenizer.encode_plus(
            t2,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
i1['input_features'] = [10.0, 2.0, 0.020512820512820513, 0, 30.0, -0.008333333333333331]
i2['input_features']  = [15.5, 2.5, 0.016443850267379677, 0, 18.725806451612904, 0.2]
saliency_map = generate_saliency_map(model, i1, i2)
print(saliency_map[0], saliency_map[1])