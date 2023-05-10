from whatsapp_to_db import WhatsAppToDb
from langchain.llms.base import LLM
from langchain import SQLDatabaseChain, SQLDatabase

DATABASE_PATH = "data/whatsapp_export.db"


class SqlQA:
    def __init__(
        self, model: LLM, input_chat_file_path: str, parse: bool = False
    ) -> None:
        # Read a CSV and create a db.
        if parse:
            WhatsAppToDb().parse_txt_to_db(input_chat_file_path, DATABASE_PATH)
        self.model = model
        database_handle = SQLDatabase.from_uri(f"sqlite:///{DATABASE_PATH}")
        self.llm_chain = SQLDatabaseChain.from_llm(
            self.model, database_handle, verbose=True
        )

    def answer(self, question: str) -> str:
        return self.llm_chain.run(question)
