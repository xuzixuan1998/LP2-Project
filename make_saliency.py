import torch
from transformers import AutoTokenizer
from finetune import FTLogReg

def generate_saliency_map(model, input_tokens):
    # Set model in evaluation mode
    model.eval()

    # Convert input tokens to tensor
    ids, masks, features = input_tokens['input_ids'].unsqueeze(0), input_tokens['attention_mask'].unsqueeze(0), torch.tensor(input_tokens['input_features']).unsqueeze(0)

    # Enable gradient calculation for the input tensor
    ids.requires_grad_()
    features.requires_grad_()
    # Forward pass to get model predictions
    output = model(ids, masks, features)

    # Calculate gradients
    model.zero_grad()
    output.sum().backward()  # Backward pass

    # Get the gradients of the input tensor
    ids_gradients = ids.grad[0]
    features_gradients = features.grad[0]
    # Normalize gradients
    ids_gradients = torch.abs(ids_gradients)
    ids_gradients /= ids_gradients.max()
    features_gradients = torch.abs(features_gradients)
    features_gradients /= features_gradients.max()


    return ids_gradients.detach().numpy(), features_gradients.detach().numpy()

model_name = 'bert-base-uncased'

model = FTLogReg(model_name, True, True)
state_dict = torch.load('bert-base-uncased_d2_features_finetune.pth')
model.load_state_dict(state_dict)
input_text = "Agh I was worried this might be the case. You don't think the guard telling me the things would be alright is something to pursue a case on?"
# input_text = "I am a bot whose sole purpose is to improve the timeliness and accuracy of responses in this subreddit."
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_tokens = tokenizer.encode_plus(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
input_tokens['input_features'] = [14.0, 1.5, 0.03982683982683983, 1, 42.285714285714285, 0.0]
# input_features = [19.0, 1.0, 0.012048192771084338, 1, 50.578947368421055, 0.0]
saliency_map = generate_saliency_map(model, input_tokens)
print(saliency_map[0], saliency_map[1])