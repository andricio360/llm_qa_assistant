# POC AWS Sagenaker Documentation Q&A System with Langchain and OpenAI

This project implements a Proof of Concept (PoC) for a Question & Answer (Q&A) system using Langchain for text processing and OpenAI for language models. The system allows users to input questions and retrieves answers from a collection of documents of AWS Sagemaker documentation.

## Overview

The Q&A system is built using Python and leverages the Langchain library for text processing tasks such as document splitting and vectorization. It utilizes OpenAI's language models for generating answers to user queries.

## Features

- Integration with Langchain and OpenAI for advanced text processing and language modeling.
- User-friendly interface implemented with Streamlit, allowing users to input questions and retrieve answers.
- Efficient document indexing and retrieval using Chroma vector stores.
- Flexible configuration options for customizing the system's behavior.

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/andricio360/llm_qa_assistant.git
    ```

2. Install dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key by creating a `.streamlit` folder in the project directory with a file inside called `secrets.toml`. Then add your API key:

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage

To run the Q&A system, execute the following command:

```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/qa_main.py
