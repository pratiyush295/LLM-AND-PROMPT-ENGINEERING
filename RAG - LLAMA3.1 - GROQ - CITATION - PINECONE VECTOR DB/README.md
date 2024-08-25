
# **Advanced Question-Answering System Using RAG LLMs**

This project is based on a tutorial from Superteams.ai and demonstrates the implementation of a complex Retrieval-Augmented Generation (RAG) system using LLaMa 2, Pinecone, Groq, and Gradio. The system retrieves relevant information from a document set and generates contextually accurate answers.

## **Table of Contents**
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Learnings](#learnings)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## **Introduction**

This project involves building a sophisticated question-answering platform by integrating multiple AI tools. It leverages state-of-the-art models and technologies to enhance the accuracy and relevance of generated responses.

## **Prerequisites**

Before you begin, ensure you have met the following requirements:
- Python 3.7+
- Google Colab or a local Python environment
- API keys for LLaMa 2, Pinecone, and Groq

## **Installation**

To get started, clone this repository and install the required libraries:

```bash
git clone https://github.com/yourusername/rag-llm-question-answering.git
cd rag-llm-question-answering
pip install llama-index llama-parse langchain-community openai pinecone-client groq gradio
```

## **Configuration**

You'll need to set up environment variables for your API keys. This can be done using the following code snippet:

```python
import os
from google.colab import userdata

os.environ['LLAMA_CLOUD_API_KEY'] = userdata.get("LLAMAPARSE")
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI')
```

## **Usage**

### **Step 1: Document Parsing and Vectorization**

First, parse the documents and convert them into vectors using LLaMa 2 and Pinecone:

```python
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(api_key="YOUR_LLAMA_API_KEY", result_type="markdown")
documents = SimpleDirectoryReader(input_files=['data/LLAMA-TEST.pdf'], file_extractor={".pdf": parser}).load_data()
```

### **Step 2: Tokenize and Prepare Metadata**

Next, tokenize the parsed documents into paragraphs and prepare the metadata for Pinecone indexing:

```python
import re

def tokenize_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [{'text': para, 'section_no': i + 1} for i, para in enumerate(paragraphs)]

metadata_documents = [{'text': page.text, 'page_no': i + 1} for i, page in enumerate(documents)]
```

### **Step 3: Generate Embeddings and Indexing in Pinecone**

Generate embeddings for each section and upsert them into Pinecone:

```python
from openai import OpenAI
openai.api_key = os.environ['OPENAI_API_KEY']

client = OpenAI()

def generate_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
index = pc.Index("chat")

vectors = []
for page in metadata_documents:
    for section in tokenize_paragraphs(page['text']):
        vectors.append({
            'id': str(len(vectors)),
            'values': generate_embedding(section['text']),
            'metadata': {'page_no': page['page_no'], 'section_no': section['section_no'], 'content': section['text']}
        })

index.upsert(vectors)
```

### **Step 4: Querying and Generating Responses**

Query the Pinecone index and generate responses using Groq:

```python
def query_pinecone(query_text):
    query_embedding = generate_embedding(query_text)
    result = index.query(vector=query_embedding, top_k=5, include_values=True, include_metadata=True)
    return [{'metadata': match['metadata'], 'content': match['metadata']['content'][0]} for match in result['matches']]

from groq import Groq

groq_client = Groq(api_key='YOUR_GROQ_API_KEY')

def generate_response(query, docs):
    system_message = "You are a helpful AI assistant. Answer the question using the provided context."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    response = groq_client.chat.completions.create(model="llama3-70b-8192", messages=messages)
    return response.choices[0].message.content
```

### **Step 5: Build the Gradio Interface**

Finally, build an interactive interface using Gradio:

```python
import gradio as gr

def build_gradio_interface():
    interface = gr.Interface(
        fn=lambda query: generate_response(query, query_pinecone(query)),
        inputs="text",
        outputs="text"
    )
    interface.launch()

if __name__ == "__main__":
    build_gradio_interface()
```

## **Learnings**

- Mastered RAG (Retrieval-Augmented Generation) systems.
- Improved skills in integrating LLMs with external systems like Pinecone and Groq.
- Enhanced proficiency in Python and API integration.

## **Acknowledgements**

This project is based on a tutorial from [Superteams.ai](https://superteams.ai/), which provided a foundational understanding of the RAG framework and its application in advanced AI systems.

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.
