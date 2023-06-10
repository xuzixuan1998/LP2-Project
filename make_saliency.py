import pdb
import torch
from transformers import AutoTokenizer
from finetune import FTLogReg

def generate_saliency_map(model, i1, i2):
    # Set model in evaluation mode
    model.eval()

    # Convert input tokens to tensor
    ids = (i1['input_ids'], i2['input_ids'])
    masks = (i1['attention_mask'], i2['attention_mask'])
    features = (torch.tensor(i1['input_features']).unsqueeze(0).requires_grad_(), torch.tensor(i2['input_features']).unsqueeze(0).requires_grad_())
    # Forward pass to get model predictions
    model.pretrain.embeddings.word_embeddings.weight.requires_grad_()
    output = model(ids, masks, features)

    # Calculate gradients
    model.zero_grad()
    output.sum().backward()  # Backward pass

    # Get the gradients of the input tensor

    embedding_gradients = model.pretrain.embeddings.word_embeddings.weight.grad
    idx1, idx2 = ids[0][ids[0] != 0], ids[1][ids[1] != 0]
    ids_gradients_1, ids_gradients_2 =torch.norm(embedding_gradients[idx1], p=2, dim=1), torch.norm(embedding_gradients[idx2], p=2, dim=1)
    features_gradients_1, features_gradients_2 = torch.abs(features[0].grad[0]), torch.abs(features[1].grad[0])
    # Normalize gradients
    ids_gradients_1 /= ids_gradients_1.max()
    ids_gradients_2 /= ids_gradients_2.max()
    features_gradients_1 /= features_gradients_1.max()
    features_gradients_2 /= features_gradients_2.max()


    return (ids_gradients_1.detach().numpy(), ids_gradients_2.detach().numpy(), features_gradients_1.detach().numpy(), features_gradients_2.detach().numpy())

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
print(saliency_map)