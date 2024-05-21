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








def get_text_from_uploaded_files(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename.endswith(".pdf"):
            with uploaded_file as pdf_file:  # No need for open()
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        elif filename.endswith(".csv"):
            with uploaded_file as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    text += "\n".join(row)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            text += df.to_string(index=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    return text




def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(predictions,user_name):

    prompt_template = """
    you are expert to write a mail to the student regarding the following query.
    Mentioned the ticket category and provide the answer to the student query.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    if you did not get anything in context then you are also expert to send a automated mail to the student regarding the Ticket Category.
    for example:- Query: I want to take an admission in Phd\n
    Answer: Hope you are doing well! We are glad to know that you are interested in taking admission in Phd. Our team will get in touch with you to assist you further.\n
    if Context is not there provide something like we will reach you soon\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Ticket Category: \n{predictions}\n

    in below format

    Hi {user_name}, Good Day!

    Hope you are doing well!.

    Ticket Category : {predictions}\n

    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n

    Regards,\n
    Jain University 
    
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question","predictions","user_name"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question,predictions,user_name,student_email):
    if "history" not in st.session_state:
        st.session_state.history = []

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(predictions,user_name)

    # Maintain a history of last 5 questions and answers
    history = st.session_state.history
    if len(history) > 5:
        history.pop(0)  # Remove the oldest entry if history exceeds 5 items

    response = chain(
        {"input_documents":docs, "question": user_question,"predictions":predictions,"user_name":user_name}
        , return_only_outputs=True)
    
    print(response)
    st.subheader(":red[*Response:*]", divider='rainbow')
    st.write(response["output_text"])

    history.append({"question": user_question, "answer": response["output_text"]})
    
    mail.send_email_to_student(student_email, response["output_text"])
    # Display the history
    st.subheader(":red[*Recent Interactions:*]", divider='rainbow')
    for item in history:
        st.write(f"[*Question*]: {item['question']}")
        st.write(f"[*Answer*]: {item['answer']}\n")
    return response["output_text"]







def main():
    st.set_page_config("InquiryIQ")
    st.header("Jain University Inquiry Team 💁 ")

    col1, col2 = st.columns([1, 1])
    with col1:
        user_name = st.text_input("Name", value="")
    with col2:
        # Get the student's email address input
        student_email = st.text_input("Enter your email", "")

   

# Validate the email format (optional)
    if student_email and not re.match(r"[^@]+@[^@]+\.[^@]+", student_email):
        st.warning("Please enter a valid email address.")#"nitinkumar78302609@gmail.com"

    
    user_question = st.text_input("Ask your Query here:")
    submit = st.button("Submit")
    
    
    if submit:
        


        # Load the trained model from joblib file
        model = joblib.load("model.pkl")
        vec = joblib.load("vectorizer.pkl")
        new_texts = [user_question]

    # Vectorize the new texts using the loaded vectorizer
        new_texts_vectorized = vec.transform(new_texts)

    # Make predictions using the loaded model
        predictions = model.predict(new_texts_vectorized)
        user_input(user_question,predictions,user_name,student_email)
    # Display the predictions
        print(predictions)
    
        
    with st.sidebar:
        st.title("Upload Data:")
        pdf_docs = st.file_uploader("Upload your Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_uploaded_files(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



# write a function to send an email to the student with the answer to the query




if __name__ == "__main__":
    main()