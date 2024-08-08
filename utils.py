# load libraries
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st


openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

from pinecone import Pinecone, ServerlessSpec
import os
os.environ[
    "PINECONE_API_KEY"
] = "9ca3ab29-e230-4616-94de-d542214a8ffd"

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)

index = pc.Index('chatbotcs', host='https://chatbotcs-3ronozs.svc.aped-4627-b74a.pinecone.io')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# def query_refiner(conversation, query):

#     response = openai.Completion.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string      