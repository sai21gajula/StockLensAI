from test_1 import *
import streamlit as st

st.title("Capstone Early Prototype")
st.header("LLM RAG-Based Chatbot for Stock & Finance Q&A with Price Trend Prediction")
st.header("Stocks/ Finance")

if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

if "nq" not in st.session_state:
    st.session_state.nq = []

qqq = st.text_input("Enter your question:", st.session_state.question)

col1, col2 = st.columns(2)

with col1:
    if st.button("Submit"):
        if qqq.strip():
            output = graph.invoke({
                "question": qqq,
            })
            st.session_state.answer = output["answer"]
            st.session_state.question = qqq
            st.session_state.nq = output["followup_questions"]
        else:
            st.session_state.answer = "Please ask a question."

with col2:
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

st.write(st.session_state.answer)

st.write("Recommended Questions:")
for question in st.session_state.nq:
    st.write(question)