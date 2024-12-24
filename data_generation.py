from dotenv import load_dotenv
from prompt_template import PromptTemplate

import streamlit as st
import json
import os
from collections import Counter
import altair as alt
import pandas as pd


import google.generativeai as genai


import pandas as pd


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to load Google Gemini Pro Vision API And get response
def get_gemini_response(Question, Answer):
    model = genai.GenerativeModel('gemini-pro')
    template = """
    You are expert to clissify the data into different categories like Simple, Average or Complex.
    you can use string matching, keyword extraction, or any other method to classify the data.

    Use this Question {Question} and Answer {Answer} to classify the data into different complexity level with grade 1 - 10.

    Output should have only integer value like 1, 2, 3 ,4, etc:

    

    """

    # Create an instance of PromptTemplate

    prompt = PromptTemplate(input_variables=["Question","Answer"], template=template)

    filled_prompt = prompt.fill_template(Question=Question, Answer=Answer)

    print(filled_prompt)
    
    response = model.generate_content(filled_prompt)
    #print(response.text)
    return response.text

# Initialize an empty list to store the generated records
records = []

try:
    # Generate the specified number of records
    df = pd.read_excel("Book.xlsx")
    #df = df1[df1["Ticket Category"]== "Career Guidance"]
    # Extract questions and answers
    questions = df["Student message"]
    answers = df["Handler message"]
    for i in range(len(df)):
        try:
            #print(questions[i], answers[i])
            category = get_gemini_response(questions[i], answers[i])
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
    df.to_excel("classified_file.xlsx", index=False)
except Exception as e:
    print(e)


print("CSV file saved successfully!")


    