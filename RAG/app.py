import sys
print(sys.executable)

import streamlit as st
from librarian import get_query_engine

st.set_page_config(page_title="DS Librarian", layout="centered")
st.title("📚 Data Science Librarian")

# Initialize the engine once and store it in the session
if "query_engine" not in st.session_state:
    with st.spinner("Initializing Local Brain..."):
        st.session_state.query_engine = get_query_engine()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your toolkit..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # This calls your local Llama 3!
        response = st.session_state.query_engine.query(prompt)
        st.markdown(str(response))
        st.session_state.messages.append({"role": "assistant", "content": str(response)})