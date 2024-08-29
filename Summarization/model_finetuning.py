import warnings
warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from Email_Dataset import EmailDataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler

# Set paths to your CSV files
body_csv_path = 'email_thread_details.csv'
summary_csv_path = 'email_thread_summaries.csv'

# Define the tokenizer and model
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Initialize the dataset
email_dataset = EmailDataset(body_csv_path, summary_csv_path)
torch_dataset = email_dataset.to_torch_dataset(tokenizer)

# Create DataLoader
data_loader = DataLoader(torch_dataset, batch_size=4, shuffle=True)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(data_loader) * 3  # For 3 epochs
scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 5
accumulation_steps = 8  # Number of steps to accumulate gradients


for epoch in range(10):  # Number of epochs
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(data_loader)}")

# Save the fine-tuned model
model.save_pretrained('fine-tuned-t5')
tokenizer.save_pretrained('fine-tuned-t5')
