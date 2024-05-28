from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from typing import List
import fitz  
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    #remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Function to preprocess and get embeddings for text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()  # Convert to numpy array

# Function to calculate cosine similarity between two sets of embeddings
def calculate_cosine_similarity(embeddings1, embeddings2):
    return cosine_similarity(np.array(embeddings1), np.array(embeddings2))

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/calculate_similarity/")
async def calculate_similarity(job_description: str = Form(...), resumes: List[UploadFile] = File(...)):
    try:
        # Process job description string
        job_descriptions = [job_description]

        # Read resumes from the uploaded PDF files
        resume_texts = []
        for resume in resumes:
            resume_text = extract_text_from_pdf(await resume.read())
            resume_texts.append(resume_text)

        results = []

        for job_description in job_descriptions:
            job_description_embeddings = get_embeddings(job_description)

            top_resumes = []
            top_similarity_scores = []

            for resume_text in resume_texts:
                resume_embeddings = get_embeddings(resume_text)
                similarity_score = calculate_cosine_similarity(job_description_embeddings, resume_embeddings)
                top_resumes.append(resume_text)  # Append the full text for now
                top_similarity_scores.append(similarity_score[0][0])

            top_indices = np.argsort(top_similarity_scores)[-3:][::-1]
            top_resumes = [top_resumes[i] for i in top_indices]
            top_similarity_scores = [top_similarity_scores[i] for i in top_indices]

            results.append({
                "job_description": job_description,
                "selected_resumes": ', '.join([resume[:200] for resume in top_resumes]),  # Truncate for display
                "similarity_scores": ', '.join(map(str, top_similarity_scores))
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
