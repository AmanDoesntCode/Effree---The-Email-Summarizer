from O365 import Account
from O365.utils.token import FileSystemTokenBackend
import pandas as pd

credentials = ('a2e45fa4-f770-4f9f-9271-a7d7fa9b83dc', 'Y1O8Q~x6OLVLgju~5dE5gemFFb1awpgDu4B9bbs8')
tk = FileSystemTokenBackend(token_path=r"D:\Programming Files\Innovation Competition\Authenetication Key", token_filename="o365_token.txt")
account = Account(credentials,tenant_id="2c5bdaf4-8ff2-4bd9-bd54-7c50ab219590",token_backend=tk)

if not account.is_authenticated:
    # This will open a browser window for interactive authentication if necessary
    url = account.authenticate(scopes=['basic', 'mailbox'])

mailbox = account.mailbox()
inbox = mailbox.inbox_folder()

def clean_email_body(email_body):
    # Define the pattern to match the separator lines
    email_body = email_body.replace("__","")
    email_body = email_body.replace(" ­","")
    start = email_body.find("“")
    end = email_body.rfind("Inspiration")
    email_body = email_body.replace(email_body[start:end+len("Inspiration")],"")
    return email_body

def get_mail_dataframe():
    subjects = []
    bodies = []
    links = []
    senders = []
    ids = []
    for message in inbox.get_messages(limit=30):
        if not message.is_read:
            subjects.append(message.subject)
            email_body = message.get_body_text()
            bodies.append(clean_email_body(email_body))
            links.append(message.web_link)  # Get the web link of the email
            senders.append(message.sender.address)  # Get the sender's email address
            ids.append(message.object_id)  # Get the message ID
    df = pd.DataFrame({
        'Subject': subjects,
        'Body': bodies,
        'Link': links,
        'Sender': senders,
        'ID': ids
    })
    return df

def mark_as_read(message_id):
    message = inbox.get_message(message_id)
    message.mark_as_read()
    
    
mail_df = get_mail_dataframe()
mail_df.to_csv('Data\Input.csv', index=False) 