import os
from google.colab import userdata
import re
import nest_asyncio
import openai
from openai import OpenAI
from pinecone import Pinecone
from groq import Groq
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from llama_parse import LlamaParse
from langchain.vectorstores import Pinecone
import pinecone


# Setup
os.environ['LLAMA_CLOUD_API_KEY'] = userdata.get("LLAMAPARSE")
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI')



nest_asyncio.apply()

parser = LlamaParse(
    api_key="LLAMA PARSE API KEY",
    result_type="markdown"
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['data/LLAMA-TEST.pdf'], file_extractor=file_extractor).load_data()
# print(len(documents))
metadata_documents = [{'text': page.text, 'page_no': documents.index(page) + 1} for page in documents]

openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI()

from pinecone import Pinecone
pc = Pinecone(api_key="PINE CONE API KEY")
index = pc.Index("chat")

groq_client = Groq(api_key='GROQ API KEY')

# Functions
def tokenize_paragraphs(text):
    send = []
    count = 1
    paragraphs = re.split(r'\n\s*\n', text.strip())
    for para in paragraphs:
        send.append({'text': para, 'section_no': count})
        count += 1
    return send

def metadata_documents_section():
    super_document_metadata = []
    for metadata_page in metadata_documents:
        document_metadata = {}
        page_metadata = []

        page_no = metadata_page['page_no']
        rt = tokenize_paragraphs(metadata_page['text'])
        for section in rt:
            temp = {}
            temp['section'] = section['text']
            temp['section_number'] = section['section_no']
            page_metadata.append(temp)
        document_metadata['page_no'] = page_no
        document_metadata['data'] = page_metadata
        super_document_metadata.append(document_metadata)
    return super_document_metadata

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def query_pinecone(query_text):
    query_embedding = generate_embedding(query_text)
    result = index.query(vector=query_embedding, top_k=1, include_values=True, include_metadata=True)

    matches = []
    for match in result['matches']:
        matches.append({
            'metadata': {'page_no': match['metadata']['page_no'], 'section_no': match['metadata']['section_no']},
            'content': match['metadata']['content'][0]
        })

    return matches

def collect_content(retrieved_docs):
    rt = []
    for content in retrieved_docs:
        rt.append(content['content'])
    return rt

def generate_response(query, only_content, docs):
    system_message = (
        "You are a helpful AI assistant. Answer the question using the provided context.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(only_content))
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return response.choices[0].message.content



metadata = metadata_documents_section()

id = 0
vectors = []
for page in metadata:
    page_no = page['page_no']
    for section in page['data']:
        temp = {}
        temp['id'] = str(id)
        temp['values'] = generate_embedding(section['section'])
        temp['metadata'] = {'page_no': page_no, 'section_no': section['section_number'], 'content': [section['section']]}
        vectors.append(temp)
        id += 1
index.upsert(vectors)

# index.query(vector=generate_embedding("when was japan attacked?"), top_k=2, include_values=False, include_metadata=True)










def processing(query):
        retrieved_docs = query_pinecone(query)
        metadata = retrieved_docs[0]['metadata']
        only_content = collect_content(retrieved_docs)
        answer = generate_response(user_query, only_content, retrieved_docs)
        print(answer)
        return str(answer)+'\n\n Context From : \n'+f"Page No : {metadata['page_no']}\tSection No : {metadata['section_no']}"

demo = gr.Interface(
    fn=processing,
    inputs=["text"],
    outputs=["text"],
)




if "__name__"==__main_:
    user_query = "who was hitler"
    retrieved_docs = query_pinecone(user_query)
    metadata = retrieved_docs[0]['metadata']
    only_content = collect_content(retrieved_docs)
    answer = generate_response(user_query, only_content, retrieved_docs)
    demo.launch(debug=True)
