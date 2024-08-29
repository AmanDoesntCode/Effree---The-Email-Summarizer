import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Email_Dataset import EmailDataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
# Initialize model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(r'D:\Programming Files\Innovation Competition\Summarization\fine-tuned-t5')  # Update with your model path
model = T5ForConditionalGeneration.from_pretrained(r'D:\Programming Files\Innovation Competition\Summarization\fine-tuned-t5')  # Update with your model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the dataset
email_dataset = EmailDataset('email_thread_details.csv', 'email_thread_summaries.csv')
dataset = email_dataset.to_torch_dataset(tokenizer)
dataset_size = len(dataset)
train_size = int(0.98 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Evaluate the model
model.eval()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rougeL_scores = []

for batch in tqdm(dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    predictions = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate ROUGE scores
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

# Calculate average ROUGE scores
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-L: {avg_rougeL:.4f}")


