from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from dotenv import load_dotenv
import streamlit as st
import json
import os
from collections import Counter
import altair as alt
import pandas as pd
import transformers
import torch

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGING_FACE_API_KEY")

from huggingface_hub import login

login(token="")


# Use a pipeline as a high-level helper
from transformers import pipeline

llama_pipeline = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")


#llama_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# Load LLaMA model and tokenizer
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
# llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# model_id = "meta-llama/Meta-Llama-3-8B"

# llama_pipeline = transformers.pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )

# Function to get response from LLaMA model
def get_llama_response(question, answer):
    template = """
    You are an expert at classifying data into different categories like Simple, Average, or Complex.
    You can use string matching, keyword extraction, or any other method to classify the data.

    Use this Question: {question} and Answer: {answer} to classify the data into different complexity levels with grades 1 - 10.

    The output should have only integer values like 1, 2, 3, 4, etc.
    """

    filled_prompt = template.format(question=question, answer=answer)
    print(filled_prompt)

    # Generate response using the LLaMA pipeline
    response = llama_pipeline(filled_prompt, max_length=50, num_return_sequences=1)
    response_text = response[0]['generated_text'].strip()
    print(response_text)

    # Extract the first integer from the response text
    try:
        category = int(next(filter(str.isdigit, response_text.split())))
    except (ValueError, StopIteration):
        category = None

    return category

# Initialize an empty list to store the generated records
records = []

try:
    # Load the Excel file
    df = pd.read_excel("Book.xlsx")
    # Extract questions and answers
    questions = df["Student message"]
    answers = df["Handler message"]

    for i in range(2):
        try:
            category = get_llama_response(questions[i], answers[i])
            print(category)
            df.loc[i, "grade"] = category
            val = category
            if val is not None:
                if val <= 4:
                    df.loc[i, "Category"] = "Simple"
                elif val > 4 and val <= 7:
                    df.loc[i, "Category"] = "Average"
                else:
                    df.loc[i, "Category"] = "Complex"
                print("Category Assigned Successfully!", df['Category'])
                print(f"{i+1} Records Successfully Generated!")
            else:
                print(f"Invalid category value for record {i+1}")
        except Exception as inner_e:
            print(f"Error occurred while processing record {i+1}: {inner_e}")

    df.to_excel("classified_file_test.xlsx", index=False)
except Exception as e:
    print(e)

print("Excel file saved successfully!")
