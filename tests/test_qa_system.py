import pytest
import sys

sys.path.append(".")
from src.qa_main import QASystem

@pytest.fixture
def qa_system():
    return QASystem()

def test_answer_question(qa_system):
    question = "What is Sagemaker?"
    qa_system.build_qa_chain()
    answer = qa_system.qa_chain({"query": question})
    print(answer)
    assert isinstance(answer['result'], str)