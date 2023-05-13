from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain import PromptTemplate, LLMChain


class SimpleQA:
    def __init__(self, model: LLM) -> None:
        self.model = model
        self.prompt = self.__form_prompt()
        self.llm_chain = LLMChain(llm=self.model, prompt=self.prompt)

    def __form_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["chat", "question"],
            template="""
            Read the following transript of a chat between two or more persons.
            Answer the question below correctly. Provide an explanation for your answer.
            Where possible cite the relevant part of the transcript.
            
            {chat}

            Question: {question}
            """,
        )

    def answer(self, chat: str, question: str) -> str:
        return self.llm_chain.run(chat=chat, question=question)
