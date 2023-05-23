from bots.sql_qa import SqlQA
from bots.large_corpus_qa import LargeCorpusQA
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.agents import initialize_agent, AgentType

class E2EQA:

    def __init__(self, sqlQABot: SqlQA, largeCorpusQABot: LargeCorpusQA, model: BaseLanguageModel) -> None:
        self.sqlQABot = sqlQABot
        self.largeCorpusQABot = largeCorpusQABot
        self.model = model
        self.__initialize_agent()

    def answer(self, question: str) -> str:
        return self.agent.run(question)
    
    def __initialize_agent(self):
        self.agent = initialize_agent(
            tools=[self.__create_semantic_search_tool(), self.__create_sql_search_tool()],
            llm=self.model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def __create_semantic_search_tool(self):
        return Tool.from_function(
            name="vs_tool",
            description='''Use this tool for Semantic Similarity Questions: Fetching messages closest in meaning to the question,
            Recommendation Questions, Anomaly detection, and Summarization.
            Provide the input as a natural language question.
            ''',
            func=self.largeCorpusQABot.answer,
        )
    
    def __create_sql_search_tool(self):
        return Tool.from_function(
            name="scan_tool",
            description='''
            Use this tool for Retrieval Questions: Fetching data based on conditions,
            Aggregation Questions: Calculations over a group of rows,
            Time-Based Questions: Filtering or aggregating data based on dates or times,
            Existence or Absence Questions: Checking whether certain data exists or not,
            Pattern Matching Questions: Searching for patterns in the data,
            Ranking Questions: Ranking data based on certain criteria,
            Data Integrity Questions: Checking the consistency and validity of data,
            Statistical Questions: Performing statistical calculations on the data.
            Provide the input as a natural language question.
            ''',
            func=self.sqlQABot.answer,
        )

