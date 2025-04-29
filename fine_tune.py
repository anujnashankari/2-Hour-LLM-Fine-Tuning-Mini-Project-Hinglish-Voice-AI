# fine_tune.py - Script to fine-tune DistilBERT on Hinglish dataset
import json
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW  # Import AdamW from torch.optim instead of transformers
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Loading dataset...")

# Load the dataset.jsonl file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load dataset
dataset = load_jsonl('dataset.jsonl')
print(f"Loaded {len(dataset)} examples from dataset.jsonl")

# Extract prompts and completions
prompts = []
completions = []

for item in dataset:
    prompt = item['prompt']
    completion = item['completion']

    # Remove prefixes if they exist
    if "User: " in prompt:
        prompt = prompt.replace("User: ", "")
    if "Assistant: " in completion:
        completion = completion.replace("Assistant: ", "")

    prompts.append(prompt)
    completions.append(completion)

# Create a mapping dictionary for retrieval
prompt_completion_dict = dict(zip(prompts, completions))

# Save the mapping dictionary for inference
with open('prompt_completion_dict.json', 'w', encoding='utf-8') as f:
    json.dump(prompt_completion_dict, f, ensure_ascii=False, indent=2)

# Create positive examples (matching prompt-completion pairs)
positive_examples = []
for i in range(len(prompts)):
    positive_examples.append({
        'prompt': prompts[i],
        'completion': completions[i],
        'label': 1  # Positive example
    })

# Create negative examples (mismatched prompt-completion pairs)
negative_examples = []
for i in range(len(prompts)):
    # Select a random completion that's different from the current one
    other_indices = [j for j in range(len(completions)) if j != i]
    if other_indices:
        j = np.random.choice(other_indices)
        negative_examples.append({
            'prompt': prompts[i],
            'completion': completions[j],
            'label': 0  # Negative example
        })

# Combine positive and negative examples
all_examples = positive_examples + negative_examples
np.random.shuffle(all_examples)

# Split into training and validation sets
train_examples, val_examples = train_test_split(all_examples, test_size=0.2, random_state=42)

print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Custom dataset class
class HinglishDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example['prompt']
        completion = example['completion']
        label = example['label']

        # Tokenize the text pair
        encoding = self.tokenizer(
            prompt,
            completion,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Remove the batch dimension
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = HinglishDataset(train_examples, tokenizer)
val_dataset = HinglishDataset(val_examples, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Define the model
class HinglishClassifier(nn.Module):
    def __init__(self):
        super(HinglishClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(768, 2)  # 768 is the hidden size of DistilBERT, 2 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation
        logits = self.classifier(cls_output)
        return logits

# Initialize the model
model = HinglishClassifier()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
print(f"Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Create directory for saving model
os.makedirs('model', exist_ok=True)

# Save the model
model_path = 'model/hinglish_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save the tokenizer
tokenizer_path = 'model/tokenizer'
tokenizer.save_pretrained(tokenizer_path)
print(f"Tokenizer saved to {tokenizer_path}")

# Create embeddings for all prompts for faster inference
print("Creating embeddings for all prompts...")

def get_embedding(text, tokenizer, model, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding

# Get embeddings for all prompts
prompt_embeddings = {}
for prompt in prompts:
    prompt_embeddings[prompt] = get_embedding(prompt, tokenizer, model, device)

# Save embeddings
np.savez_compressed('model/prompt_embeddings.npz', embeddings=prompt_embeddings)
print("Embeddings saved to model/prompt_embeddings.npz")

print("Fine-tuning completed successfully!")