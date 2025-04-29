# inference.py - Script to test the fine-tuned model
import json
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model and data for inference...")

# Define the model class (same as in fine_tune.py)
class HinglishClassifier(nn.Module):
    def __init__(self):
        super(HinglishClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('model/tokenizer')

# Initialize the model
model = HinglishClassifier()

# Load the saved model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model/hinglish_model.pt', map_location=device))
model.to(device)
model.eval()

# Load the prompt-completion dictionary
with open('prompt_completion_dict.json', 'r', encoding='utf-8') as f:
    prompt_completion_dict = json.load(f)

# Load the pre-computed embeddings
data = np.load('model/prompt_embeddings.npz', allow_pickle=True)
prompt_embeddings = data['embeddings'].item()

# Function to get embedding for a new query
def get_embedding(text, tokenizer, model, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding

# Function to find the most similar prompt
def find_most_similar_prompt(query, prompt_embeddings, tokenizer, model, device):
    query_embedding = get_embedding(query, tokenizer, model, device)

    similarities = {}
    for prompt, embedding in prompt_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)[0][0]
        similarities[prompt] = similarity

    return max(similarities.items(), key=lambda x: x[1])[0]

# Function to generate a response
def generate_response(query, prompt_embeddings, prompt_completion_dict, tokenizer, model, device):
    most_similar_prompt = find_most_similar_prompt(query, prompt_embeddings, tokenizer, model, device)
    return prompt_completion_dict[most_similar_prompt]

# Test prompts
test_prompts = [
    "Kal ka plan kya hai?",
    "Kuch khaas news?",
    "Coffee peene chalein?"
]

# Generate and display responses
print("\nTesting the Hinglish assistant model with sample prompts:\n")
test_results = []

for prompt in test_prompts:
    print(f"User: {prompt}")
    response = generate_response(prompt, prompt_embeddings, prompt_completion_dict, tokenizer, model, device)
    print(f"Assistant: {response}")
    print()
    test_results.append({"prompt": prompt, "response": response})

# Save test results
with open('test_results.json', 'w', encoding='utf-8') as f:
    json.dump(test_results, f, ensure_ascii=False, indent=2)

print("Test results saved to test_results.json")

# Interactive mode
print("\nEnter 'quit' to exit")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break

    response = generate_response(user_input, prompt_embeddings, prompt_completion_dict, tokenizer, model, device)
    print(f"Assistant: {response}")

print("Inference completed.")