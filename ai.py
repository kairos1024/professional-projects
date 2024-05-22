from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from google.colab import files
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Upload the JSON file
uploaded = files.upload()

# Get the file name
json_file_name = next(iter(uploaded))

# Load JSON file
def load_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# Extract dialogue pairs from JSON data
def extract_dialogue_pairs(data):
    dialogue_pairs = []
    for entry in data:
        if 'role' in entry and 'message' in entry:
            role = entry['role']
            message = entry['message']
            intent = entry.get('intent', None)
            resources = entry.get('resources', None)
            dialogue_pairs.append((role, message, resources, intent))
    return dialogue_pairs

# Tokenize dialogue pairs, resources, and intent classifiers
def tokenize_dialogue_pairs(dialogue_pairs, tokenizer):
    tokenized_dialogues = []
    tokenized_resources = []
    tokenized_intents = []
    for role, message, resources, intent_classifier in dialogue_pairs:
        tokenized_dialogues.append(tokenizer.encode(message, truncation=True, max_length=512))
        if resources:
            tokenized_resources.append([tokenizer.encode(resource['title'], truncation=True, max_length=512) for resource in resources])
        if intent_classifier:
            tokenized_intents.append(tokenizer.encode(intent_classifier, truncation=True, max_length=512))
    return tokenized_dialogues, tokenized_resources, tokenized_intents

# Load JSON file
data = load_json_file(json_file_name)

# Extract dialogue pairs from JSON data
dialogue_pairs = extract_dialogue_pairs(data)

# Initialize tokenizer
model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to device
model = model.to(device)

# Tokenize dialogue pairs, resources, and intent classifiers
tokenized_dialogues, tokenized_resources, tokenized_intents = tokenize_dialogue_pairs(dialogue_pairs, tokenizer)

# Print tokenized dialogue pairs, resources, and intent classifiers for inspection
print("Tokenized Dialogue Pairs:")
for idx, tokenized_dialogue in enumerate(tokenized_dialogues):
    print(f"Tokenized Dialogue Pair {idx + 1}: {tokenized_dialogue}")
print()
print("Tokenized Resources:")
for idx, tokenized_resource_list in enumerate(tokenized_resources):
    print(f"Tokenized Resource List {idx + 1}: {tokenized_resource_list}")
print()
print("Tokenized Intents:")
for idx, tokenized_intent in enumerate(tokenized_intents):
    print(f"Tokenized Intent {idx + 1}: {tokenized_intent}")
print()

# Additional processing...
# Training, fine-tuning, etc.

# Extract user queries, oasis responses, and intent classifiers from parsed_dataset
user_queries = []
oasis_responses = []
intent_classifiers = []
for entry in data:
    if 'User' in entry:
        user_queries.append(str(entry['User']))
        oasis_responses.append(str(entry.get('Oasis', '')))
        intent_classifier = entry.get('Intent Classifier', None)
        if intent_classifier is not None:
            intent_classifiers.append(intent_classifier)

print(f"Size of user_queries: {len(user_queries)}")
print(f"Size of oasis_responses: {len(oasis_responses)}")
print(f"Size of intent_classifiers: {len(intent_classifiers)}")

tokenized_user_queries = []
for query in user_queries:
    if isinstance(query, str) and query.strip():
        print(f"Query: {query}")
        try:
            tokenized_user_queries.append(tokenizer.encode(query, truncation=True, max_length=512))
        except ValueError as e:
            print(f"ValueError: {e} for Query: {query}")

tokenized_oasis_responses = []
for response in oasis_responses:
    if isinstance(response, str) and response.strip():
        print(f"Response: {response}")
        try:
            tokenized_oasis_responses.append(tokenizer.encode(response, truncation=True, max_length=512))
        except ValueError as e:
            print(f"ValueError: {e} for Response: {response}")

if not tokenized_dialogues:
    print("Error: tokenized_dialogues is empty.")
else:
    train_dialogues, val_dialogues = train_test_split(tokenized_dialogues, test_size=0.2, random_state=42)

    # Convert lists to tensors and pad sequences
    train_inputs = pad_sequence([torch.tensor(d) for d in train_dialogues], batch_first=True)
    val_inputs = pad_sequence([torch.tensor(d) for d in val_dialogues], batch_first=True)

    # Move tensors to the device
    train_inputs = train_inputs.to(device)
    val_inputs = val_inputs.to(device)

    # Print size of train and validation inputs
    print(f"Size of train_inputs: {train_inputs.size()}")
    print(f"Size of val_inputs: {val_inputs.size()}")




total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")



import time
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import TensorDataset, DataLoader
from peft import prepare_model_for_kbit_training
from peft import LoraConfig
from peft import get_peft_model
from transformers import get_linear_schedule_with_warmup

# Define your dataset and dataloader
train_dataset = TensorDataset(train_inputs)  # Assuming train_inputs is defined
validation_dataset = TensorDataset(val_inputs)  # Assuming val_inputs is defined
batch_size = 8  # Define your desired batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Check requires_grad for model parameters
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Define hyperparameters for LoRA configuration
lora_hyperparameters = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": [
        'o_proj',
        'qkv_proj',
        'attn',
        'layernorm',
        'ffn',
    ],
    "bias": "none",
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM",
}

# Apply LoRA to the model with specified hyperparameters
config = LoraConfig(**lora_hyperparameters)
lora_model = get_peft_model(model, config)
for name, param in lora_model.named_parameters():
    print(name, param.requires_grad)

# Define optimizer with weight decay
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-5, weight_decay=0.01)

# Define learning rate scheduler
num_epochs = 3  # Define the number of training epochs
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 10% of training steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Define custom data collator
class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, device='cpu'):
        super().__init__(tokenizer)
        self.device = device

    def __call__(self, features):
        if isinstance(features[0], dict):
            # Use default collation if features are dictionaries
            return super().__call__(features)
        else:
            batch = {}
            if features:
                if isinstance(features[0], (tuple, list)):
                    # Handle case where features list contains a tuple or list
                    batch["input_ids"] = torch.stack([item[0] for item in features]).to(self.device)
                    if len(features[0]) > 1:
                        batch["attention_mask"] = torch.stack([item[1] for item in features]).to(self.device)
                    batch["labels"] = batch["input_ids"].clone()
                    return batch
                else:
                    # Handle case where features list contains individual tensors
                    batch["input_ids"] = features[0].to(self.device)
                    batch["attention_mask"] = features[1].to(self.device) if len(features) > 1 else None
                    batch["labels"] = batch["input_ids"].clone()
                    return batch
            else:
                # Handle case where features list is empty
                return {}

# Use custom data collator with the tokenizer
data_collator = CustomDataCollator(tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=100,
    num_train_epochs=num_epochs,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True
)

# Define Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator  # Pass the custom data collator
)

# Train the model
start_time = time.time()
trainer.train()
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print(f"Training completed in {training_time} seconds.")

model_path = "phi_3_oasis"
trainer.save_model(model_path)

# Check requires_grad for inputs
for name, param in trainer.model.named_parameters():
    print(name, param.requires_grad)

trainer.push_to_hub("kairos1024/phi-3-oasis")