# Effree — The Email Summarizer

Effree is an intelligent email summarization system that uses transformer-based NLP models to condense lengthy emails into concise, actionable summaries. The project is designed to improve productivity for users handling high-volume inboxes by combining email ingestion, preprocessing, model inference, and structured summary delivery in a scalable pipeline.

# Screenshots

<img width="1396" height="786" alt="image" src="https://github.com/user-attachments/assets/daa8bd97-0a89-4039-89e0-0e2c0981f1bb" />

<img width="1399" height="789" alt="image" src="https://github.com/user-attachments/assets/2a13d5ab-6a7a-46a6-9583-2893736e72df" />

## Features

- Summarizes long emails into short, readable outputs
- Reduces information overload for high-volume inboxes
- Supports structured preprocessing and NLP-based cleaning
- Designed for integration with enterprise email workflows
- Built with extensibility for future providers like Gmail and Zoho Mail
- Modular pipeline for ingestion, preprocessing, summarization, and output delivery

## Problem It Solves

Professionals often spend too much time reading long emails, repeated threads, and verbose business communication. Effree helps by generating compact summaries so users can quickly understand:
- the core message
- important action items
- priority information
- overall context of the email

## Tech Stack

**Core Technologies**
- Python
- NLP / Transformers
- T5 / fine-tuned summarization models
- ETL pipeline design

**Possible Integrations**
- Microsoft Outlook
- Gmail
- Zoho Mail

**Supporting Tools**
- FastAPI or Flask
- Hugging Face Transformers
- Pandas / preprocessing utilities
- REST APIs

## System Architecture

```text
Email Source -> Ingestion Layer -> Preprocessing / Cleaning -> Summarization Model -> Output Formatter -> User Interface / API
```

## Pipeline Overview

1. **Email Ingestion**
   - Connects to mailbox or imported email source
   - Fetches raw email content and metadata

2. **Preprocessing**
   - Cleans signatures, formatting noise, and repeated thread text
   - Normalizes email body for downstream summarization

3. **Summarization**
   - Uses a transformer-based model to generate concise summaries
   - Can be adapted for domain-specific business email styles

4. **Delivery**
   - Returns summaries through UI, API, or future enterprise integrations

## Project Goals

- Improve productivity for users with crowded inboxes
- Build a reusable summarization workflow for enterprise communication
- Create a scalable architecture for email intelligence applications
- Enable future extensions like action item extraction, sentiment detection, and priority scoring

## Use Cases

- Business email summarization
- Executive inbox management
- Customer support inbox compression
- Productivity tools for professionals
- Internal communication summarization

## Example Output

### Input Email
> Hi team  
> We’ve reviewed the client’s feedback from yesterday’s meeting. They want the dashboard revised before Friday, especially the KPI cards and export workflow. Please coordinate with design and backend teams and send a draft by Thursday evening. Also, make sure the cost estimate is updated before the next review.

### Summary
- Client requested dashboard revisions before Friday
- Main changes: KPI cards and export workflow
- Coordinate with design and backend teams
- Send draft by Thursday evening
- Update cost estimate before next review

## Repository Structure

```bash
Effree/
│── data/                # Datasets / processed email samples
│── notebooks/           # Experiments and model exploration
│── src/                 # Core source code
│   ├── ingestion/       # Email fetching / connectors
│   ├── preprocessing/   # Cleaning and normalization
│   ├── models/          # Summarization model code
│   ├── pipeline/        # End-to-end workflow
│   └── utils/           # Helper functions
│── app/                 # Web app / API layer
│── requirements.txt     # Python dependencies
│── README.md
```

## Installation

```bash
git clone https://github.com/AmanDoesntCode/Effree---The-Email-Summarizer.git
cd Effree---The-Email-Summarizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Project

### If it is a Python app
```bash
python app.py
```

### If it is a FastAPI app
```bash
uvicorn app.main:app --reload
```

### If it is a Streamlit app
```bash
streamlit run app.py
```

## Configuration

Create a `.env` file if your project uses API keys or email credentials:

```env
EMAIL_PROVIDER=outlook
EMAIL_USER=your_email_here
EMAIL_PASSWORD=your_password_here
MODEL_NAME=t5-small
API_KEY=your_api_key_here
```

## Future Improvements

- Multi-provider inbox support
- Thread-aware summarization
- Action item extraction
- Priority classification
- Sentiment and intent analysis
- Calendar/task integration
- Fine-tuned enterprise summarization models

## Resume-Friendly Project Summary

Effree is an intelligent email summarization system that uses transformer-based NLP pipelines to compress lengthy business emails into concise, actionable summaries. It is designed to improve inbox productivity through scalable ingestion, preprocessing, inference, and delivery workflows.

## Contributing

Contributions, suggestions, and improvements are welcome.

```bash
fork -> clone -> create branch -> commit -> push -> open pull request
```

## License

Specify your license here, for example:

**MIT License**

## Author

**Aman Singh**  
GitHub: [AmanDoesntCode](https://github.com/AmanDoesntCode)
