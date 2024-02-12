import pytest
import sys

sys.path.append(".")
from src.qa_main import QASystem

@pytest.fixture
def qa_system():
    return QASystem()

def test_load_documents(qa_system):
    qa_system.load_documents()
    assert len(qa_system.docs) > 0

def test_split_documents(qa_system):
    qa_system.load_documents()
    qa_system.split_documents()
    assert len(qa_system.splits) > 0

def test_create_vector_db(qa_system):
    qa_system.load_documents()
    qa_system.split_documents()
    qa_system.create_vector_db()
    assert qa_system.vectordb is not None
