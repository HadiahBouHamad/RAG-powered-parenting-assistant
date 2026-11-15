import unittest

from rag import create_chat_handler


class FakeRetrievalQA:
    def __init__(self):
        self.invocations = []

    def invoke(self, inputs):
        self.invocations.append(inputs)
        return {"result": "Answer", "source_documents": []}


class ChatGreetingTests(unittest.TestCase):
    def test_greeting_skips_retrieval(self):
        retrieval_qa = FakeRetrievalQA()
        chat = create_chat_handler(retrieval_qa)

        response = chat("Hi!")

        self.assertIn("parenting documents", response)
        self.assertEqual(retrieval_qa.invocations, [])

    def test_greeting_with_a_question_still_uses_retrieval(self):
        retrieval_qa = FakeRetrievalQA()
        chat = create_chat_handler(retrieval_qa)

        response = chat("Hi, what is positive parenting?")

        self.assertEqual(response, "Answer")
        self.assertEqual(
            retrieval_qa.invocations,
            [{"query": "Hi, what is positive parenting?"}],
        )


if __name__ == "__main__":
    unittest.main()
