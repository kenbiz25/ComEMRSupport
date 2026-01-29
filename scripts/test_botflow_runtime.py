import os, sys
repo_root = os.path.abspath(os.path.join(os.getcwd()))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from rag.flow import BotFlow

class MockFaiss:
    def __init__(self):
        self.dim = 1536
    def search(self, embedding, top_k=5):
        return [{'text': 'doc1', 'score': 0.9}, {'text': 'doc2', 'score': 0.8}]

class MockLLM:
    def generate_response(self, user_message, faiss_docs):
        return 'This is a mocked answer.'

class MockWhatsApp:
    def __init__(self):
        self.sent = []
    def send_message(self, user_id, text):
        print(f"Sending to {user_id}: {text}")
        self.sent.append((user_id, text))

bf = BotFlow(MockFaiss(), MockLLM(), MockWhatsApp())
bf.handle_message('user123', 'How do I reset my password?')
print('Done')
