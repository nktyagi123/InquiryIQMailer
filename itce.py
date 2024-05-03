import pandas as pd
import numpy as np
import spacy
import re


def filter_data(data):
    df = data.rename(columns = {'Ticket Category':'label'}, inplace = True)
    print(data.shape)
    filtered_df = data[['Student message','label']]
    min_samples = 1000
    min_samples2 = 600
    min_samples1 = 350
    df1 = filtered_df[filtered_df.label == "Exams"].sample(min_samples,random_state=2022)
    df5 = filtered_df[filtered_df.label == "LMS"].sample(min_samples,random_state=2022)
    df7 = filtered_df[filtered_df.label == "General"].sample(min_samples,random_state=2022)
    df10 = filtered_df[filtered_df.label == "Fee Related"].sample(min_samples,random_state=2022)
    df2 = filtered_df[filtered_df.label == "Data and Documents"].sample(min_samples,random_state=2022)
    df3 = filtered_df[filtered_df.label == "Refund"].sample(min_samples,random_state=2022)
    df4 = filtered_df[filtered_df.label == "Certificates"].sample(min_samples2,random_state=2022)
    df6 = filtered_df[filtered_df.label == "Convocation"].sample(min_samples2,random_state=2022)
    df8 = filtered_df[filtered_df.label == "Career Guidance"].sample(min_samples2,random_state=2022)
    df9 = filtered_df[filtered_df.label == "Enrollment"].sample(min_samples1,random_state=2022)

    df_balanced = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis = 0)
    return df_balanced
    

   
def clean_text(text):
    '''
    This method generates a clean text and returns a dataframe
    '''
    text = re.sub(r'(\d{3})-(\d{3})-(\d{4})',' ', text)
    text = re.sub(r'(\d{2})/(\d{2})',' ', text)
    text = re.sub(r'\S*@\S*\s?',' ', text)
    text = re.sub(r'\s+',' ', text)
    text = re.sub(r'\'',' ', text)
    text = re.sub(r'<.*?>',' ', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*',' ', text)
    text = re.sub(r'[^0-9a-z #+_]',' ', text)
    text = re.sub(r'\(.*?\)',' ', text)
    text = re.sub(r'(\w+\.\w+\.com)',' ', text)
    text = re.sub(r'\w+\d+\w+|\w+\d+|\d+\w+',' ', text)
    text = re.sub(r'\d+|\@|\*|\=|\+|\$|\%|\.\.|\#|-|\[|\]|\?|\"',' ', text)
    text = re.sub(r'[ \n]+',' ', text)
    return text.strip().lower()
    
def another_main():
    data = pd.read_excel("tickets.xlsx")
    df_balanced = filter_data(data)
    df_balanced['Student message'] = df_balanced['Student message'].astype(str)
    df_balanced['cleaned_txt'] = df_balanced['Student message'].apply(clean_text)
    return df_balanced

