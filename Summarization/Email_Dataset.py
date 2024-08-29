# Email_dataset.py

import pandas as pd
from torch.utils.data import Dataset

class EmailDataset:
    def __init__(self, body_csv_path, summary_csv_path):
        self.body_df = pd.read_csv(body_csv_path)
        self.summ_df = pd.read_csv(summary_csv_path)
        
        self.body_df['body'] = self.body_df['body'].astype(str)
        self.body_df['subject'] = self.body_df['subject'].astype(str)
        self.body_df['thread_id'] = self.body_df['thread_id'].astype(int)

        self.body_df['body'] = self.body_df['body'].apply(self.et_details_clean_text)

        self.summ_df['summary'] = self.summ_df['summary'].astype(str)
        self.summ_df['thread_id'] = self.summ_df['thread_id'].astype(int)

        self.df = pd.merge(self.body_df, self.summ_df, on='thread_id')

    def et_details_clean_text(self, text):
        '''
        Args : str
        Function : Performs Text Cleaning
        Return : str
        '''
        text = text.replace('-----Original Message-----\nFrom: ', '')
        text = text.replace('=09', '')
        text = text.replace('=20', '')
        text = text.replace('\n', ' ')
        return text.strip()

    def to_torch_dataset(self, tokenizer):
        return EmailTorchDataset(self.df, tokenizer)

class EmailTorchDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        body = item['body']
        summary = item['summary']
        
        # Tokenize the inputs and targets
        inputs = self.tokenizer(body, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        targets = self.tokenizer(summary, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        
        # Convert tensors to standard format
        inputs_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()

        return {'input_ids': inputs_ids, 'attention_mask': attention_mask, 'labels': labels}
