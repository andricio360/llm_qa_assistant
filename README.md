# POC AWS Sagemaker Documentation Q&A System with Langchain and OpenAI

This project implements a Proof of Concept (PoC) for a Question & Answer (Q&A) system using Langchain for text processing and OpenAI for language models. The system allows users to input questions and retrieves answers from a collection of documents of AWS Sagemaker documentation.

## PoC Architecture
This PoC is implemented to deploy using streamlit following this basic architecture

<img width="407" alt="image" src="https://github.com/andricio360/llm_qa_assistant/assets/54078516/b92d85f0-64b3-4e04-a15d-d582a274ab3f">

## Final Deployment Architecture
The final system will be hosted in AWS using AWS services such as Sagemaker, API Gateway, Lambda, etc. This is the overall architecture of the final solution:

<img width="641" alt="image" src="https://github.com/andricio360/llm_qa_assistant/assets/54078516/c3a7675a-5784-4a92-b564-ada13ce36dc7">

## Process Diagram
The following is the process diagram of the system functionality:

<img width="529" alt="image" src="https://github.com/andricio360/llm_qa_assistant/assets/54078516/b696af37-4712-4db0-b6a9-bd77fc7bb994">

### Step by Step explanation:

- Markdown documents are loaded into the data folder.
- Those documents are ingested using langchain and splitted into chunks.
- Those chunks are transformed into embeddings and saved into a Vector DB
- When the question arrives, it is transformed into embedding and compared with similar documents inside VectorDB
- The query and similar documents are joined into a single "prompt" as context and asks an OpenAI model to generate a response
- The final answer is then deployed into the streamlit app where the user can ask the questions and retrieve the answers from the documentation.

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
2. Switch to the branch where you'll find the latest changes:

    ```bash
    git switch develop_source_documents
    ```

3. Install dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up your OpenAI API key by creating a `.streamlit` folder in the project directory with a file inside called `secrets.toml`. Then add your API key:

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
```

## Example

![Captura de Pantalla 2024-02-18 a la(s) 8 23 41 p  m](https://github.com/andricio360/llm_qa_assistant/assets/54078516/2e71d0a5-f0b3-492d-98cb-f8d8e1cac8a8)

