# Job.Query : Job and Resume matching with chatbot

## Overview

This project is designed to streamline the recruitment process by providing a job and resume matching system and a chatbot for applicants. The key functionalities include:

1. **Job and Resume Matching**: This feature calculates a matching score between a resume and a job description, enabling HR to rank resumes based on their relevance to the job requirements.
2. **Applicant Chatbot**: A chatbot feature for applicants that answers general queries about the job portal, such as the number of available jobs.

## Features

### 1. Job and Resume Matching

- **Process Flow**:
  - **Extract**: Data extraction from resumes and job descriptions.
  - **Clean**: Preprocessing the text data (removing stop words, punctuation, etc.).
  - **Tokenize**: Splitting text into individual words or tokens.
  - **Embed**: Converting tokens into numerical vectors using embedding techniques.
  - **Cosine Similarity**: Calculating the cosine similarity between job description vectors and resume vectors.
  - **Similarity Score**: Generating a similarity score to rank resumes.

- **Technologies Used**:
  - Python
  - Natural Language Processing (NLP)
  - Cosine Similarity for measuring the similarity between documents

### 2. Applicant Chatbot

- **Functionality**:
  - Applicants can query the job portal for general information (e.g., number of available jobs).
  - Utilizes LangChain SQL Agent for handling queries and retrieving data from the database.

- **Technologies Used**:
  - LangChain
  - SQL for database queries
  - Python for chatbot implementation


### 3. Result 

- **Job and Resume Matching**
![Alt text](/media/image2.png)


- **Chatbot**

![Alt text](/media/image.png)
