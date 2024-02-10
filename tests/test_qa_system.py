import pytest
from qa_system import QASystem

@pytest.fixture
def qa_system():
    return QASystem()

def test_answer_question(qa_system):
    question = "What is Sagemaker?"
    answer = qa_system.answer_question(question)
    assert isinstance(answer, str)
    assert len(answer) > 0