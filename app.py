import streamlit as st
import pandas as pd
from email_reader import get_mail_dataframe, mark_as_read
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import style
import st_cs

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(r'D:\Programming Files\Innovation Competition\Summarization\fine-tuned-t5')
model = T5ForConditionalGeneration.from_pretrained(r'D:\Programming Files\Innovation Competition\Summarization\fine-tuned-t5')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def summarize_email(email_text, max_input_length=512, max_output_length=150):
    inputs = tokenizer.encode("summarize: " + email_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs, max_length=max_output_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st_cs.set_background('6393101.jpg')
# Add custom CSS for white background and card styling
st.markdown(style.CUSTOM_CSS, unsafe_allow_html=True)
# Display the image instead of the title and description
st.image("Screenshot 2024-07-30 202020.png", use_column_width=True)

st.markdown(f"""
<div class="rouge-metrics">
    <h4>Summary Metrics</h4>
    <p>Avg ROUGE-1: {0.5481}</p>
    <p>Avg ROUGE-L: {0.4603}</p>
</div>""", unsafe_allow_html=True)

with st.spinner('Loading emails and generating summaries...'):# Get email data
    mail_df = get_mail_dataframe()

    cols = st.columns(2)

    for index, row in mail_df.iterrows():
        summary = summarize_email(row['Body'])
        col = cols[index % 2]
        # Render the email card
        with col:
            st.markdown(f"""
            <div class='card'>
                <h3>{row['Subject']}</h3>
                <p>{summary}</p>
                <a href="{row['Link']}">Read the full email</a>
                <div class='sender'>{row['Sender']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render the checkbox and handle its action
            checkbox_key = f"mark_as_read_{index}"
            if st.checkbox("mark as read", key=checkbox_key):
                mark_as_read(row['ID'])
                st.success(f"Email from {row['Sender']} marked as read")
        
    # Check for marked emails and call mark_as_read
