import streamlit as st
from simple_qa import SimpleQA

st.header("Personalizing Question-Answering Models")
# The first tab will be a no-memory simple question answering demo.
[basic_qa_tab] = st.tabs(["Basic QA"])

with basic_qa_tab:
    chat_input = st.text_area(
        label="Enter a small chat transcript here",
        max_chars=10000,
        placeholder="""
                [7/9/18, 6:53:47 PM] Jane Doe: Ok bye , will call in an hour
                [7/9/18, 7:54:09 PM] John Doe: Did you call?
                [7/9/18, 8:01:09 PM] Jane Doe: Yes, I did.
                  """,
    )
    question_input = st.text_input(
        label="Enter a question here", placeholder="Did Jane call John?"
    )
    qa_bot = SimpleQA()
    if st.button("Answer"):
        st.write(qa_bot.answer(chat=chat_input, question=question_input))
