from dotenv import load_dotenv
from openai import AzureOpenAI
import openai
import streamlit as st
from tenacity import retry, wait_exponential, stop_after_attempt
import os
from openai import AzureOpenAI

# load_dotenv()
# Load environment variables
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# deployment = "gpt-4o"
st.set_page_config(page_title="Azure OpenAI Chat", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'config'

def go_to_chat(client):
    st.session_state.client = client
    st.session_state.page = 'chat'
    st.rerun()

def ask(prompt:str, client:AzureOpenAI) :
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful friend.",
            },
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        max_completion_tokens=200,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=st.session_state.deployment
    )
    return response

def config_page():
    st.set_page_config(page_title="Config App")
    st.title("Config the Chat")

    if 'messages' not in st.session_state:
        st.session_state.messages = [{"api_key": "", "deployment": "", "azure_endpoint": " "}]
    if 'api_key' not in st.session_state:
        st.session_state.api_key = " "
    if 'deployment' not in st.session_state:
        st.session_state.deployment = " "
    if 'azure_endpoint' not in st.session_state:
        st.session_state.azure_endpoint = " "

    st.session_state.api_key = st.text_input("Inserisci la tua API key:")
    st.session_state.deployment = st.text_input("Inserisci il deployment:")
    st.session_state.azure_endpoint = st.text_input("Inserisci l'azure endpoint:")
    if st.button("Send"):

    #ipotizzo che alla creazione vengano controllate le varie configurazioni
        try:
            client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=st.session_state.azure_endpoint,
                api_key=st.session_state.api_key,
            )
            
            simple_prompt = {"role": "user", "content": "Rispondimi se vedi questo messaggio."}
            ask(simple_prompt, client=client)
            st.write("Connessione Riuscita.")
            go_to_chat(client)

        except Exception as e :
            st.error(f"Connessione non Riuscita: {e}")
            st.rerun()


def chat_page(client):
    st.set_page_config(page_title="Chat App")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.title("Simple Chat")

    user_input = st.text_input("Inserisci un messaggio:")

    if st.button("Send"):
        #salva il messaggio dell'utente
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            response = ask(user_input,client)

            assistant_response = response.choices[0].message.content

            #mostro a video la risposta
            st.write(f"{assistant_response}")

            #salvo il messaggio del modello
            st.session_state.messages.append({"role":"assistant", "content":assistant_response})
        except Exception as e :
            st.error(f"{e}")

    # if st.button("Show History Chat"):
    #     for messages in st.session_state.messages:
    #         if messages["role"] == "user":
    #             st.write(messages["content"])

# response = ask()
# print(response.choices[0].message.content)

def main():
    # Render the appropriate page
    if st.session_state.page == 'config':
        config_page()
    elif st.session_state.page == 'chat':
        client = st.session_state.client
        chat_page(client)

if __name__ == "__main__":
    main()