import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(r'D:\Programming Files\Innovation Competition\Summarization\fine-tuned-t5')
model = T5ForConditionalGeneration.from_pretrained(r'D:\Programming Files\Innovation Competition\Summarization\fine-tuned-t5')

# Ensure the model is set to evaluation mode
model.eval()

# If you have a GPU, move the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def summarize_email(email_text,max_input_length=3084,  max_output_length=250):
    # Tokenize the input text
    # inputs = tokenizer.encode("summarize :" + email_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    inputs = tokenizer.encode("summarize :" + email_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    inputs = inputs.to(device)
    
    # Generate the summary
    summary_ids = model.generate(inputs, max_length=max_output_length, num_beams=25, length_penalty=2.0, early_stopping=True)
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example email text
email_text = """
Dear Learners,A new week is the perfect opportunity to sharpen your skills and stay competitive in your career. Engage in weekly sessions guided by professionals to enhance your knowledge and stay up to date.SESSION DETAILS: JULY 29 – AUGUST 2 Session TopicSession DetailsSpeaker DetailsGenerative AI Landscape on AWSLearn AWS and unlock the transformative power of Artificial Intelligence.July 29, 2024  |  5:00 PM – 6:00 PMMs. Prajila VK(Senior Principal Consultant- Learning, Infosys)Fostering Emotional Intelligence in the WorkplaceEnhance your understanding of emotional intelligence to manage emotions effectively resulting to building stronger relationships, and promoting a positive work environment.July 30, 2024  |  5:00 PM – 6:00 PMMr. Vivek C(Product Manager & Solution Architect, IIHT)Building Self-EsteemEnhance your self-awareness, promote positive self-talk, and learn practical strategies to boost confidence and resilience.July 31, 2024  |  5:00 PM – 6:00 PMMr. Aman Girdhar(Senior Analyst - Learning, Infosys)  Introduction to Artificial IntelligenceThis course introduces Artificial Intelligence using various use cases and real-life examples.August 1, 2024  |  5:00 PM – 6:00 PMNo Speaker session scheduledClick here to complete the course!Introduction to Deep LearningThis course highlights the importance of Deep Learning and its impact on today’s world.August 2, 2024  |  5:00 PM – 6:00 PMNo Speaker session scheduledClick here to complete the course! Webinar DetailsWebinar number2517 246 4068Webinar passwordInfy@SB2024 (46391722 from phones and video systems)Click HERE to join the sessions See you at the sessions!Regards,Team Infosys SpringboardScan the QR code to download the Infosys Springboard AppApp StorePlay StoreFollow us on  Facebook  |  Instagram  |  LinkedIn  |  X (Twitter)For queries, write to springboard-support@infosys.com
"""
# Generate and print the summary
summary = summarize_email(email_text)
print("Summary:", summary)
