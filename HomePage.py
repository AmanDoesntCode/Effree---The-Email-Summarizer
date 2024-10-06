import torch
import Design.st_cs as st_cs
import Design.style as style
import numpy as np
import streamlit as st
import Utils.email_display_utils as Dutils
from Utils.email_reader import mark_as_read
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class EmailApp:
    
    
    '''
    
    A class that majorly oversees the functioning of the Homepage of our streamlit webapp menu.
    ATTRIBUTES : 
        constructor (init) : Instantiates the pretrained Tokenizer and Model of all-MiniLM-L6-v2 since,
                             for semantic search, we need a model that can encode text into fixed-size vectors,
                             which the T5 model isnt suitable.
        
        set_background_and_style()  : sets the background and defines the styling markdown for the webpage
        
        QuerySearch() : Used to perform semantic search on the inbox's text contents and then call display_emails() 
                        with top_searches as its arguementself.
        
        _get_vector() : essentially transforms the text into a format that allows the search algorithm to
                        measure similarity between the input query and documents in a database.
                        
        display_metric() : displays a card on the webpage showing the evaluation metrics for the finetuned
                           summarization task.
                           
        display_emails() : creates a card to display the email summary along with the subject, sender, complete
                          mail link and a checkbox to mark as readself.
        
        run () : used to call most of the above functions and supply raw files like images and dataframes as the 
                 input arguments.                       
    
    ''' 
    
    
    def __init__(self):
        
        '''
        Creates an object to instantiate the pre-traied and fine-tuned T5 model and Tokenizer. 
        ARGS :  none
        RETURN :    none
        
        '''

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        
    def set_background_and_style(self, background_img, css):
        
       
        '''
        ARGS : 
                backgorund_img : text of the file path of the background image. 
                css : referencing another user-defined python module that contains the css code for styling
                    the webpage
            
            RETURN :    none
        
        ''' 
        # Set background image and custom CSS
        st_cs.set_background(background_img)
        st.markdown(css, unsafe_allow_html=True)
        

    def QuerySearch(self, summary_df):
        '''
        Performs Semantic Search on the summaries generated to addon a function of searching in the WebApp and calls
        display_emails(top_searches) to finally complete the search.  
        
        ARGS :
                summary_df : A referenced dataframe from the Display Utils module, that stores different information
                             of the unread mails in the inbox with the summary instead of the bodyself.
         
        RETURN :    None
        
        '''
        
        # we use the Subjects and the Summaries only for the Searching
        email_subjects = summary_df['Subject']
        email_summaries = summary_df['Summary']
        
        search_query = st.text_input("Search emails")
        
        if not search_query:
            # keep top indices empty if there is no search triggered
            top_indices=[]

        if search_query:
            # Pre-compute vectors for efficiency
            subject_vectors = []
            summary_vectors = []

            for text in email_subjects:
                # uses dense vectors of tokenized text to perform searching algoritm on and appends them
                vector = self._get_vector(text)
                subject_vectors.append(vector)

            for text in email_summaries:
                vector = self._get_vector(text)
                summary_vectors.append(vector)

            # Compute similarity scores once
            query_vector = self._get_vector(search_query)
            subject_similarities = cosine_similarity([query_vector], subject_vectors).flatten()
            summary_similarities = cosine_similarity([query_vector], summary_vectors).flatten()
            # Combine scores efficiently using numpy operations
            combined_scores = np.add(subject_similarities, summary_similarities) / 2
            # Find top matches
            top_indices = np.argsort(combined_scores)[-6:][::-1]

            st.write("Top Matching Emails:")
        
        self.display_emails(summary_df,top_indices)


    def _get_vector(self, text):
        
        '''
        Essentially transforms the text into a format that allows the search algorithm to
        measure similarity between the input query and documents in a database.
        
        ARGS:
                text : recieves any type of text be it summaries, subjects or search queries.
        
        RETURN: A NumPy array representing the mean-pooled embedding of the input text for semantic similarity.
                
        '''
        
        
        
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**encoding)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
            
    def display_metrics(self, avg_rouge_1, avg_rouge_l):
        '''
        Display Summary metrics (hard-coded here but calculated seperately) in a card with the html class "rouge-metrics"
        
        ARGS : 
                avg_rouge_1 : Average of ROUGE-1 scores of different batches of the summariation task
                avg_rouge_l : Average of ROUGE-L scores of different batches of the summarization task
        
        RETURN :    None
        
        '''
             
        st.markdown(f"""
        <div class="rouge-metrics">
            <h4>Summary Metrics</h4>
            <p>Avg ROUGE-1: {avg_rouge_1}</p>
            <p>Avg ROUGE-L: {avg_rouge_l}</p>
        </div>""", unsafe_allow_html=True)                 
    
    
    
    def display_emails(self,df, top_indices):
        
        '''
        The main component of the HomePage. Displays Email summaries and details in a card on the WebApp.
        
        Extra ATTRIBUTES :
                mark_as_read : called from email_reader module. marks an email read. 
    
        ARGS:
                df :  dataframe recieved from Display_utils module with the summary and other details.
                top_indices : recieves it from SearchQuery() to check if the search is done or not.
        
        RETURN :    None    
        
        '''
        
        # if there is no search query display the cards normally in a 2 column design.  
        if top_indices != []:
            
            cols = st.columns(1)
            for index in top_indices:
                row = df.iloc[index]
                col = cols[index % 1]  # Distribute across columns
                
                # Display email card
                with col:
                    st.markdown(f"""
                    <div class='Scard'>
                        <h3>{row['Subject']}</h3>
                        <p>{row['Summary']}</p>
                        <a href="{row['Link']}">Read the full email</a>
                        <div class='sender'>{row['Sender']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    checkbox_key = f"mark_as_read_{index}"
                    if st.checkbox("Mark as read", key=checkbox_key):
                        mark_as_read(row['ID'])
                        st.success(f"Email from {row['Sender']} marked as read")
                 
        
        # if search query has been triggered the arrangement of cards shifts to 1 column
        else:    
            
            cols = st.columns(2)
            for index, row in df.iterrows():
                summary = row['Summary']
                col = cols[index % 2]
                
                # Display email card
                with col:
                    st.markdown(f"""
                    <div class='card'>
                        <h3>{row['Subject']}</h3>
                        <p>{summary}</p>
                        <a href="{row['Link']}">Read the full email</a>
                        <div class='sender'>{row['Sender']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display the checkbox to mark as read
                    checkbox_key = f"mark_as_read_{index}"
                    if st.checkbox("mark as read", key=checkbox_key):
                        mark_as_read(row['ID'])
                        st.success(f"Email from {row['Sender']} marked as read")

    def run(self):
        # Orchestrate the flow of the app
        st.image(r"Artifacts\Screenshot 2024-07-30 202020.png", use_column_width=True)
        self.display_metrics(0.5481, 0.4603)  # Static metrics as placeholders
        self.set_background_and_style(r'Artifacts\6393101.jpg', style.CUSTOM_CSS)
        with st.spinner('Loading emails and generating summaries...'):
            summary_df = Dutils.get_summaryDF()
            self.QuerySearch(summary_df)
            
        
