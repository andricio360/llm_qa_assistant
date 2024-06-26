import glob
import os
import sys
from typing import List, Any, Dict, Union

import langchain
import numpy as np
import openai
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings

sys.path.append("../..")

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

class QASystem:
    """Q&A System class."""

    def __init__(self) -> None:
        """Initialize QASystem."""
        self.markdown_folder: str = "./data/"
        #self.persit_directory: str = "docs/chromadb/"
        self.llm_name: str = "gpt-3.5-turbo"
        self.temperature: int = 0
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        self.docs: List[str] = []
        self.vectordb = None
        self.llm = None
        self.qa_chain = None
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]

    def load_documents(self) -> None:
        """Load documents from markdown files."""
        markdown_paths = glob.glob(self.markdown_folder + "*.md")
        for path in markdown_paths:
            loader = UnstructuredMarkdownLoader(path)
            self.docs.extend(loader.load())

    def split_documents(self) -> None:
        """Split documents into chunks."""
        self.splits = self.text_splitter.split_documents(self.docs)

    def create_vector_db(self) -> None:
        """Create vector database."""
        self.vectordb = Chroma.from_documents(
            documents=self.splits,
            embedding=OpenAIEmbeddings(openai_api_key=self.openai_api_key),
            #persist_directory=self.persist_directory,
        )

    def create_llm_model(self) -> None:
        """Create language model."""
        self.llm = ChatOpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
        )

    def retrieve_documents(self) -> None:
        """Retrieve documents from VectorDB."""
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.vectordb.as_retriever()
        )

    def build_qa_chain(self) -> None:
        """Build Q&A chain."""
        self.load_documents()
        self.split_documents()
        self.create_vector_db()
        self.create_llm_model()
        self.retrieve_documents()


class QASystemApp:
    """Q&A System application."""

    def __init__(self, qa_system: QASystem) -> None:
        """Initialize QASystemApp."""
        self.qa_system = qa_system

    def run_streamlit_app(self) -> None:
        """Run Streamlit app."""
        self.qa_system.build_qa_chain()
        st.title("AWS Sagemaker Documentation Q&A Assistant 👩‍💻")
        question = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            result = self.qa_system.qa_chain({"query": question})
            answer = result["result"]
            st.write("Answer:", answer)


if __name__ == "__main__":
    qa_system = QASystem()
    qa_system_app = QASystemApp(qa_system)
    qa_system_app.run_streamlit_app()

