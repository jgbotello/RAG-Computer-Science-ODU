from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import emoji

# Create a header with the title of the app
st.subheader("CSChat")

# Two variables (responses and requests) are checked if they are present in the Streamlit session state. If they are not present, they are initialized with default values.
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi There! I am John AI. I was designed to Help Dr. Weigle to answer your questions about anything related to the CS Handbook. How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# initialize conversational model
llm = ChatOllama(model_name="mistral:7b-instruct-v0.2-fp16", api_key=os.getenv("OLLAMA_API_KEY"))

# Conversation memory configuration
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3, return_messages=True)

# Message templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="Answer the question as truthfully as possible using your knowledge and the provided context")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

def display_message(text, is_user=False):
    if is_user:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: flex-end;">
            <div style="background-color: #DCF8C6; padding: 10px; border-radius: 10px; max-width: 70%;">
                {text}
            </div>
            <span style="font-size: 1.5em; margin-left: 10px;">{emoji.emojize(':boy:')}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5em; margin-right: 10px;">{emoji.emojize(':robot:')}</span>
            <div style="background-color: #F1F0F0; padding: 10px; border-radius: 10px; max-width: 70%;">
                {text}
            </div>
        </div>
        """, unsafe_allow_html=True)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:

    query = st.text_input("Query: ", key="input")

    if query:
        with st.spinner("typing..."):
            context = find_match(query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)


with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            display_message(st.session_state['responses'][i], is_user=False)
            if i < len(st.session_state['requests']):
                display_message(st.session_state["requests"][i], is_user=True)
