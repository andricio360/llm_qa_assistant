import pytest
import sys
import os
from streamlit.testing.v1 import AppTest
sys.path.append(".")
from src.qa_main import QASystem

@pytest.fixture
def qa_system():
    at = AppTest.from_file("src/qa_main.py", default_timeout=1000)
    at.secrets["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    at.run()
    print(at.main)
    return QASystem()

def test_load_documents(qa_system):
    qa_system.load_documents()
    assert len(qa_system.docs) > 0

def test_split_documents(qa_system):
    qa_system.load_documents()
    qa_system.split_documents()
    assert len(qa_system.splits) > 0