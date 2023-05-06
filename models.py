from langchain.chat_models import ChatOpenAI
from langchain.llms.replicate import Replicate


def get_openai_model():
    return ChatOpenAI(temperature=0)


def get_replicate_model():
    return Replicate(
        model="replicate/vicuna-13b:a68b84083b703ab3d5fbf31b6e25f16be2988e4c3e21fe79c2ff1c18b99e61c1",
        input={
            "temperature": 0.01,
        },
        verbose=True,
    )
