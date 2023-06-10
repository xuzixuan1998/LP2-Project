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
    print(output)
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
    # features_gradients_1 /= features_gradients_1.max()
    # features_gradients_2 /= features_gradients_2.max()


    return (ids_gradients_1.detach().numpy(), ids_gradients_2.detach().numpy(), features_gradients_1.detach().numpy(), features_gradients_2.detach().numpy())

model_name = 'bert-base-uncased'

model = FTLogReg(model_name, True, True)
state_dict = torch.load('bert-base-uncased_d2_features_finetune.pth')
model.load_state_dict(state_dict)
t1 = "Also, for the folks unaware and all “war is terrible”: the russian war machine made the spectacularly sound decision to place live ammunitions right next to the bunker 😑."
t2 = "Whether it’s laziness or sheer incompetence, moscow did more to kill all 450+ soldiers than Ukraine did."
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.tokenize(t1))
print(tokenizer.tokenize(t2))
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
i1['input_features'] = [29.0, 1.0, 0.007407407407407408, 2, 53.0, -0.02541743970315399]
i2['input_features']  = [17.0, 3.0, 0.02531645569620253, 1, 46.41176470588235, 0.25]
saliency_map = generate_saliency_map(model, i1, i2)
print(saliency_map)