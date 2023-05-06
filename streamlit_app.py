import streamlit as st
from simple_qa import SimpleQA
from large_corpus_qa import LargeCorpusQA
from models import get_openai_model
import logging

logging.basicConfig(level=logging.INFO)

st.header("Personalizing Question-Answering Models")
# The first tab will be a no-memory simple question answering demo.
[basic_qa_tab, memory_tab] = st.tabs(["Basic QA", "QA with memory"])


@st.cache_resource
def get_simple_qa_bot():
    return SimpleQA(get_openai_model())


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
        key="simple_qa_question",
        label="Enter a question here",
        placeholder="Did Jane call John?",
    )
    qa_bot = get_simple_qa_bot()
    if st.button("Answer", key="simple_qa_answer"):
        st.write(qa_bot.answer(chat=chat_input, question=question_input))


# This will use a single instance of the LargeCorpusQA class for all users.
@st.cache_resource
def get_large_corpus_qa_bot():
    return LargeCorpusQA(get_openai_model())


with memory_tab:
    question_input = st.text_input(
        key="large_corpus_qa_question",
        label="Enter a question here",
        placeholder="Did Jane call John?",
    )

    large_corpus_qa_bot = get_large_corpus_qa_bot()
    if st.button(
        "Load",
        key="large_corpus_qa_load",
        help="This will take a few minutes. Click Load and come back after a coffee break.",
    ):
        large_corpus_qa_bot.initialize_vector_store(None)
    if st.button("Answer", key="large_corpus_qa_answer"):
        st.write(large_corpus_qa_bot.answer(question_input))
