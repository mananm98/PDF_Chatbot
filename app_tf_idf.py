#!/usr/bin/env python3.11
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.retrievers import TFIDFRetriever
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
import pickle
import os
from dotenv import load_dotenv
import sklearn

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
    This is an LLM powered chatbot
                ''')
    add_vertical_space(5)


def main():
    st.header("Chat with PDF!")
    pdf = st.file_uploader("Upload your PDF",type = "pdf")
    load_dotenv()
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name[:-4]
        

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        # Create Chunks
        chunks = text_splitter.split_text(text=text)
        list_of_chunks = []
        for txt in chunks:
            txt = txt.replace("\t"," ")
            if len(txt) < 20:
                continue
            list_of_chunks.append(txt)
        
        # for i,v in enumerate(list_of_chunks):
        #     print(i,v)
        #     print("🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎🥎")
        # print(len(list_of_chunks))
        
        retriever = TFIDFRetriever.from_texts(list_of_chunks,k=3)

        if os.path.exists(f'{pdf_name}.pkl'):
            with open(f'{pdf_name}.pkl','rb') as f:
                Vectordb = pickle.load(f)
            st.write("Embeddings loaded form the disk")

        else:
            # Create embedding unction
            embedding_function = OpenAIEmbeddings()
            Vectordb = FAISS.from_texts(chunks,embedding_function)
            with open(f'{pdf_name}.pkl','wb') as f:
                pickle.dump(Vectordb,f)
            st.write("Embeddings computed !")

        # query the db
        query = st.text_input("ask questions about to PDF")
        
        if query:
            # docs = Vectordb.similarity_search(query,k = 3)
            docs = retriever.get_relevant_documents(query)
            st.write(docs)
            prompt_template = """Assume you are the business owner and you have hired a consultant to improve your business, use the following pieces of context of your business to answer the questions asked by the consultant. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""
            PROMPT = PromptTemplate(
                                    template=prompt_template, input_variables=["context", "question"]
                                )
            
            llm = OpenAI(temperature = 0,model_name="gpt-3.5-turbo",openai_api_key = st.secrets["OPENAI_API_KEY"])
            # prompt.format(docs=docs)
            # PROMPT.format(input_documents= docs, question= query )

            chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT, verbose=True)
            with get_openai_callback() as cb:
                response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
                # print(cb)
            st.write(response)


if __name__ == '__main__':
    main()