
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI , Request , Form
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
import openai
import os

from fastapi import FastAPI, UploadFile, File, Form
from PyPDF2 import PdfReader
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import yake
import torch
import nltk
from pydantic import BaseModel


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class Resume(BaseModel):
    text: str

import re

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def extract_skills_keywords(text):
    keyword_extractor = yake.KeywordExtractor()

    skill_keywords_extracted = keyword_extractor.extract_keywords(text)

    skill_keywords = [keyword for keyword, score in skill_keywords_extracted]

    return skill_keywords


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text

nltk.download('stopwords')
 


from transformers import DistilBertTokenizer, DistilBertModel

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to preprocess and get embeddings for text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()  # Convert to numpy array

@app.post("/calculate_similarity/")
async def calculate_similarity(job_description: str = Form(...), resumes: list[UploadFile] = File(...)):
    try:
        preprocessed_job_description = preprocess_text(job_description)
  
        job_description_embeddings = get_embeddings(preprocessed_job_description)
        
        results = []

        for resume in resumes:
            resume_text = extract_text_from_pdf(resume.file)
            preprocessed_resume = preprocess_text(resume_text)
            resume_embeddings = get_embeddings(preprocessed_resume)
            similarity_score = cosine_similarity(job_description_embeddings, resume_embeddings)[0][0]
            similarity_percentage = similarity_score * 100
            
            results.append({"filename": resume.filename, "similarity_score": similarity_percentage})
        
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

    
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
db = SQLDatabase.from_uri('sqlite:///job.db')
agent_executor = create_sql_agent(llm, db=db, agent_type='openai-tools', verbose=True)


@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTMLResponse(content=html_content)

@app.post("/chat", response_class=HTMLResponse)
async def chat(message: str = Form(...)):
    response = agent_executor.run(message)
    return HTMLResponse(content=response)

if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)