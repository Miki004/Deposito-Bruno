import streamlit as st
from es1.script import ask
# Set up the page configuration
st.set_page_config(page_title="Chat App")

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Simple Chat")

st.session_state.name = st.text_input("Inserisci un messaggio:")
response = ask(st.session_state.name)
st.write(response.choices[0].message.content)

