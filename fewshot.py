import streamlit as st
import pandas as pd
from prompt_template import PromptTemplate
import re
import openpyxl
import google.generativeai as genai
import os
from dotenv import load_dotenv
 
load_dotenv()
 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Read few-shot examples from Excel file
few_shot_df = pd.read_excel("fewshotsample.xlsx")
 
# Construct prompt using few-shot examples
prompt_text_example = ""
for index, row in few_shot_df.iterrows():
    prompt_text_example += f"Question: {row['Student message']}\nAnswer: {row['Handler message']}\nCategory: {row['category']}\nGrade: {row['Grading']}\n\n"
 
# Function to get Gemini response
def get_gemini_response(Question, Answer, prompt_text_example):
    model = genai.GenerativeModel('gemini-pro')
    prompt_text = """
    
You are an expert in classifying data into different categories such as Simple, Average, or Complex. You can use string matching, keyword extraction, or any other method, including applying machine learning classification techniques, to classify the data.

Given a dataset containing issues and concerns raised by students and the respective responses provided by staff on a university portal ticketing system, your task is to classify the complexity of each issue-response pair.

Please follow these steps:

Analyze the Data: Examine both the issue raised by the student and the response provided by the staff.
Apply Classification Techniques: Utilize various machine learning classification techniques to determine the complexity of the data. Techniques can include:
String Matching
Keyword Extraction
Sentiment Analysis
Natural Language Processing (NLP) Methods
Supervised and Unsupervised Learning Models
Assign a Complexity Grade: Based on your analysis, assign a complexity grade to each issue-response pair on a scale from 1 to 10, where:
1 to 4 represents a "Simple" complexity level.
5 to 7 represents an "Average" complexity level.
8 to 10 represents a "Complex" complexity level.
The output should only contain an integer value representing the complexity grade, such as 1, 2, 3, 4, etc.

Example:

Student Issue: "I am having trouble accessing my course materials online."
Staff Response: "Please try clearing your browser cache and cookies, and ensure you are using the latest version of the browser."
Output: 2

Please classify the given dataset using Student Question {Question}\n  and Handler Response {Answer} and provide the complexity grade for each issue-response pair.
 
    
    """
    prompt_text += "Given the following examples:\n\n"
    prompt_text += prompt_text_example

    #print("12345678901234567890123456789",prompt_text)
    prompt = PromptTemplate(input_variables=["Question","Answer"], template=prompt_text)

    filled_prompt = prompt.fill_template(Question=Question, Answer=Answer)

    #print("#####################################################",filled_prompt)
    response = model.generate_content(filled_prompt)
    return response.text
 
# Read main dataset from Excel
df = pd.read_excel("fewshotsample.xlsx")
 
# Generate responses for each record in main dataset
questions = df["Student message"]
answers = df["Handler message"]
#print(questions)
for i in range(len(df)):
    try:
        #print(questions[i], answers[i])
        category = get_gemini_response(questions[i], answers[i], prompt_text_example)
        print(category)
        df.loc[i, "grade"] = int(category)
        val = int(category)
        if val <= 4:
            df.loc[i, "Category"] = "Simple"
        elif val > 4 and val <= 7:
            df.loc[i, "Category"] = "Average"
        else:
            df.loc[i, "Category"] = "Complex"
        print("Category Assigned Successfully!", df['Category'])
        print(f"{i+1} Records Successfully Generated!")
    except Exception as inner_e:
        print(f"Error occurred while processing record {i+1}: {inner_e}")
 
# Save the updated main dataset
df.to_excel("output_file_final.xlsx", index=False)