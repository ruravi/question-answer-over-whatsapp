from whatsapp_to_db import WhatsAppToDb
from langchain.llms.base import LLM
from langchain import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

DATABASE_PATH = "data/whatsapp_export.db"


class SqlQA:
    def __init__(
        self, model: LLM, input_chat_file_path: str, parse: bool = False
    ) -> None:
        # Read a CSV and create a db.
        if parse:
            WhatsAppToDb().parse_txt_to_db(input_chat_file_path, DATABASE_PATH)
        self.model = model
        # Create an agent that can convert questions to SQL queries and run them.
        self.pure_sql_agent = create_sql_agent(
            llm=self.model,
            toolkit=SQLDatabaseToolkit(
                db=SQLDatabase.from_uri(f"sqlite:///{DATABASE_PATH}"),
                llm=self.model,
            ),
            verbose=True,
        )

    def answer(self, question: str) -> str:
        return self.pure_sql_agent.run(question)
