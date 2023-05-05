import unittest
from langchain.llms.fake import FakeListLLM
from simple_qa import SimpleQA


class TestSimpleQA(unittest.TestCase):
    def test_simple_qa(self):
        fake_model = FakeListLLM(responses=["I am a human.", "I am a robot."])
        sut = SimpleQA(fake_model)
        actual_answer = sut.answer(
            chat="""
        [7/9/18, 6:53:47 PM] Jane Doe: Ok bye , will call in an hour
        [7/9/18, 7:54:09 PM] John Doe: Did you call?
        """,
            question="Are you a human?",
        )
        self.assertEqual("I am a human.", actual_answer)
