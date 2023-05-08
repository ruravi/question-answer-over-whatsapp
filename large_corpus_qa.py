from typing import Optional, List
from langchain.llms.base import LLM
from langchain import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from transformers import GPT2TokenizerFast
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.document_loaders import WhatsAppChatLoader
from langchain.docstore.document import Document
import logging
import threading


class LargeCorpusQA:
    def __init__(
        self,
        model: LLM,
        options: dict = {
            "chunk_size": 100,
            "persist_directory": "vector_index",
        },
    ) -> None:
        self.model = model
        self.CHUNK_SIZE = options["chunk_size"]
        self.embeddings = HuggingFaceEmbeddings()
        self.PERSIST_DIRECTORY = options["persist_directory"]
        self.qa_bot = None
        self.qa_bot_lock = threading.Lock()

    def initialize_vector_store(self, file_path: Optional[str]) -> None:
        if self.qa_bot is not None:
            logging.warning("Vector store already initialized")
            return
        retriever = (
            self.create_new_vector_store(file_path)
            if file_path
            else self.load_vector_store()
        )
        self.qa_bot = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": self.__form_prompt_template(),
            },
        )

    # Thread-safe
    def answer(self, query: str) -> str:
        with self.qa_bot_lock:
            if self.qa_bot is None:
                raise ValueError("Vector store not initialized")
            return self.qa_bot.run(query)

    def load_vector_store(self) -> BaseRetriever:
        logging.info("Loading pre-existing vector store")
        vector_store = Chroma(
            persist_directory=self.PERSIST_DIRECTORY, embedding_function=self.embeddings
        )
        return vector_store.as_retriever()

    def create_new_vector_store(self, file_path: str) -> BaseRetriever:
        logging.info("Creating new vector store")

        unchunked_documents = self.load_documents(file_path)
        chunked_documents = self.chunk_documents(unchunked_documents)
        fresh_vector_db = self.create_db(chunked_documents)

        return fresh_vector_db.as_retriever()

    def load_documents(self, file_path: str):
        document_loader = WhatsAppChatLoader(file_path)
        unchunked_documents = document_loader.load()
        return unchunked_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=self.CHUNK_SIZE, chunk_overlap=0
        )
        return text_splitter.split_documents(documents)

    def create_db(self, documents: List[Document]):
        fresh_vector_db = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=self.PERSIST_DIRECTORY,
        )
        fresh_vector_db.persist()
        return fresh_vector_db

    def __form_prompt_template(self):
        prompt_template = """
            Read the following transcripts of several chats between two or more persons.
            Answer the question below correctly. Provide an explanation for your answer.
            Where possible cite the relevant part of the transcript.
            
            {context}

            Question: {question}
            """
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
