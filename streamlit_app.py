import streamlit as st
from bots.simple_qa import SimpleQA
from bots.large_corpus_qa import LargeCorpusQA
from bots.sql_qa import SqlQA
from bots.e2e_qa import E2EQA
from models import get_openai_model
import logging

logging.basicConfig(level=logging.INFO)

st.header("Personalizing Question-Answering Models")
# The first tab will be a no-memory simple question answering demo.
[basic_qa_tab, memory_tab, database_tab, e2e_agent_tab] = st.tabs(
    ["Basic QA", "QA with index", "QA with a database", "End to end Agent QA"]
)


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

INPUT_CHAT_DATA_FILE_PATH = "data/whatsapp_export.txt"

# This will use a single instance of the LargeCorpusQA class for all users.
@st.cache_resource
def get_large_corpus_qa_bot():
    result = LargeCorpusQA(get_openai_model())
    result.initialize_vector_store(INPUT_CHAT_DATA_FILE_PATH)
    return result

with memory_tab:
    question_input = st.text_input(
        key="large_corpus_qa_question",
        label="Enter a question here",
        placeholder="Did Jane call John?",
    )

    large_corpus_qa_bot = get_large_corpus_qa_bot()
    if st.button(
        "Load a new file",
        key="large_corpus_qa_load_new",
        help="This will take a few minutes. Click Load and come back after a coffee break.",
    ):
        large_corpus_qa_bot.initialize_vector_store(INPUT_CHAT_DATA_FILE_PATH)

    if st.button("Answer", key="large_corpus_qa_answer"):
        st.write(large_corpus_qa_bot.answer(question_input))


@st.cache_resource
def get_database_bot():
    return SqlQA(get_openai_model(), INPUT_CHAT_DATA_FILE_PATH, parse=True)


with database_tab:
    question_input = st.text_input(
        key="qa_with_database_question",
        label="Enter a question here",
        placeholder="Did Jane call John?",
    )

    database_bot = get_database_bot()
    if st.button("Answer", key="qa_with_database_answer"):
        st.write(database_bot.answer(question_input))

@st.cache_resource
def get_e2e_qa_bot():
    return E2EQA(get_database_bot(), get_large_corpus_qa_bot(), get_openai_model())

with e2e_agent_tab:
    question_input = st.text_input(
        key="e2e_qa_question",
        label="Enter a question here",
        placeholder="Did Jane call John?",
    )

    e2e_qa_bot = get_e2e_qa_bot()
    if st.button("Answer", key="e2e_qa_answer"):
        st.write(e2e_qa_bot.answer(question_input))
