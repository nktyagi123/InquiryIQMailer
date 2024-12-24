import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import csv
import joblib
import mail
import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Excel file into a DataFrame
df = pd.read_excel("test.xlsx")

# Extract questions and answers
questions = df["Student message"]
answers = df["Handler message"]

# Initialize the Gemini Pro model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)



# Function to classify the entries
def classify_entry(question, answer):
    # Classify the question
    question_classification = model(question)
    # Classify the answer
    answer_classification = model(answer)
    return question_classification, answer_classification

# Function to assign categories based on classification results
def assign_category(question_classification, answer_classification):
    # Logic to assign categories based on classification results
    # For example, you can use string matching, keyword extraction, or any other method
    
    # Dummy logic: If either question or answer is classified as "Simple", classify as "Simple"
    if "Simple" in question_classification or "Simple" in answer_classification:
        return "Simple"
    # If both question and answer are classified as "Complex", classify as "Complex"
    elif "Complex" in question_classification and "Complex" in answer_classification:
        return "Complex"
    # Otherwise, classify as "Average"
    else:
        return "Average"

# Iterate through each entry, classify, and assign category
for i in range(len(df)):
    question_classification, answer_classification = classify_entry(questions[i], answers[i])
    category = assign_category(question_classification, answer_classification)
    # Assign category to the DataFrame
    df.loc[i, "category"] = category

# Save the DataFrame with categories to a new Excel file
df.to_excel("classified_file.xlsx", index=False)
