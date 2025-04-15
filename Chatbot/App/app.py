import streamlit as st
import re
import time
import requests
import json

st.title("StockLensAI")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "auto_prompt" not in st.session_state:
    st.session_state.auto_prompt = None


def on_new_question_click(q):
    st.session_state.auto_prompt = q
    st.session_state.chat_history.append({'role': 'user', 'content': q})


user_input = st.chat_input("Enter your question here")


if st.session_state.auto_prompt is not None:
    prompt = st.session_state.auto_prompt

    st.session_state.auto_prompt = None
elif user_input:
    prompt = user_input
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
else:
    prompt = None


for msg in st.session_state.chat_history:
    st.chat_message(msg['role']).markdown(msg['content'], unsafe_allow_html=True)


if prompt:
    start_time = time.time()
    
    # Send a POST request to the local server
    response = requests.post(
        "http://127.0.0.1:8000/ask",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"question": prompt})
    )
    
    elapsed_time = time.time() - start_time
    print("Overall time", elapsed_time)

    # Check if the request was successful
    if response.status_code == 200:
        output = response.json()

        answer = output.get("output", {}).get("answer", "I couldn't find an answer for that.")
        new_q = output.get("output", {}).get("n3s_question", "").split("\n") if output.get("output", {}).get("n3s_question") else []
        list_of_links = list(dict.fromkeys(output.get("output", {}).get("links", [])))[:2]


        full_response = answer
        if list_of_links:
            links_html = "<br>".join([f"<a href='{link}' target='_blank'>{link}</a>" for link in list_of_links])
            full_response += f"<br><br>{links_html}"

        st.session_state.chat_history.append({'role': 'assistant', 'content': full_response})
        st.chat_message('assistant').markdown(full_response, unsafe_allow_html=True)

        if new_q:
            st.write("Suggested questions:")
            for i, s in enumerate(new_q):
                if s and s.strip() and re.search(r'\w', s):
                    st.button(s, key=f"btn_{i}", on_click=on_new_question_click, args=(s,))
    else:
        st.session_state.chat_history.append({'role': 'assistant', 'content': "Error: Unable to fetch data."})
        st.chat_message('assistant').markdown("Error: Unable to fetch data.", unsafe_allow_html=True)
