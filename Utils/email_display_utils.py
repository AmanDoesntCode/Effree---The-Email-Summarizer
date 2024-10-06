import torch
from Utils.email_reader import get_mail_dataframe
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = r'D:\Programming Files\Innovation Competition\Summarization\Models-T5ft'
TOKENIZER_PATH = r'D:\Programming Files\Innovation Competition\Summarization\Models-T5ft'


tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def summarize_email( email_text, max_input_length=512, max_output_length=150):
    # Perform the summarization
    inputs = tokenizer.encode("summarize: " + email_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs, max_length=max_output_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    
    
def get_summaryDF():
    summary=[]
    summary_df = get_mail_dataframe()
    for index, row  in summary_df.iterrows():
            smry = summarize_email(row['Body'])
            summary.append(smry)
    summary_df["Summary"] = summary
    summary_df.drop(["Body"],axis=1)
    summary_df.to_csv("Data\Output.csv",index = False)
    return summary_df

        

        