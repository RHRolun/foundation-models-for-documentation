import streamlit as st
import requests
import json
import os

GATEWAY_URL = os.environ.get("GATEWAY_URL")+"/question"

def get_reply(search_text: str) -> str:
    response = requests.post(GATEWAY_URL, json={"message": search_text})
    return response.json()

def main():    
    st.set_page_config(
        page_title="RedHat Chat Bot",
    )

    st.title("RedHat Chat Bot")
    search_text = st.text_input("", value="What is RHODS?")
    ask_button = st.button("Ask")

    if ask_button:
        reply = get_reply(search_text)
        st.markdown(reply)

if __name__ == "__main__":
    main()

