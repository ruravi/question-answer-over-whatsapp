import unittest
from langchain.llms.fake import FakeListLLM
from bots.simple_qa import SimpleQA
from bots.large_corpus_qa import LargeCorpusQA


class TestSimpleQA(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_model = FakeListLLM(responses=["I am a human.", "I am a robot."])
        return super().setUp()

    def test_simple_qa(self):
        sut = SimpleQA(self.fake_model)
        actual_answer = sut.answer(
            chat="""
        [7/9/18, 6:53:47 PM] Jane Doe: Ok bye , will call in an hour
        [7/9/18, 7:54:09 PM] John Doe: Did you call?
        """,
            question="Are you a human?",
        )
        self.assertEqual("I am a human.", actual_answer)

    def test_large(self):
        sut = LargeCorpusQA(self.fake_model)
        sut.initialize_vector_store(None)
        actual_answer = sut.answer("Are you a human?")
        self.assertEqual("I am a human.", actual_answer)

    def test_large_no_store(self):
        sut = LargeCorpusQA(self.fake_model)
        with self.assertRaises(ValueError):
            sut.answer("Are you a human?")

    def test_load_whatsapp_chats(self):
        sut = LargeCorpusQA(
            self.fake_model,
            options={"chunk_size": 20, "persist_directory": "test_vector_index"},
        )
        actual_unchunked_documents = sut.load_documents("data/whatsapp_export_test.txt")
        self.assertEqual(1, len(actual_unchunked_documents))

        actual_chunked_docs = sut.chunk_documents(actual_unchunked_documents)
        self.assertEqual(4, len(actual_chunked_docs))

        actual_db = sut.create_db(actual_chunked_docs)
        actual_results = actual_db.search("sunglasses", search_type="similarity")
        self.assertEqual(4, len(actual_results))
