
# Chatbot Reranker

This project implements a chatbot reranking system using advanced machine learning models and embeddings. The goal is to improve the relevance of chatbot responses by leveraging techniques like Sentence Transformers and vector databases.

## Introduction

The Chatbot Reranker project uses embeddings and vector search to rerank chatbot responses based on their relevance to the user's query. The implementation is designed to integrate with popular Python libraries like `sentence-transformers` and `qdrant-client`.

## Installation

To set up the environment and install the required dependencies, run the following command:

```bash
!pip install -q llama-index langchain-community groq gradio qdrant_client sentence-transformers
```

## Usage

1. **Import Necessary Libraries**: Begin by importing the essential libraries.

    ```python
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from google.colab import userdata
    ```

2. **Set Up Qdrant Client**: Set up the Qdrant client and import additional models.

    ```python
    from qdrant_client.models import Distance, VectorParams
    ```

3. **Login to Hugging Face**: Authenticate with Hugging Face to access the desired model.

    ```python
    from huggingface_hub import login
    login(token="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")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    ```

4. **Load and Process Data**: Load your dataset and prepare the content for processing.

    ```python
    df = pd.read_csv("hf://datasets/kdave/Indian_Financial_News/training_data_26000.csv")

    def make_content(df):
        count = 0
        data = []
        for content in df['Content']:
            temp = []
            temp.append(content)
            temp.append(df["URL"][count])
            data.append(temp)
            count += 1
        return data
    ```

5. **Initialize the Qdrant Collection**: Set up the collection within Qdrant to store and query the vectors.

    ```python
    qdrant_client = QdrantClient()
    qdrant_client.recreate_collection(
        collection_name="news_articles",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    ```

6. **Insert and Query Vectors**: Insert the content into the Qdrant collection and perform queries.

    ```python
    vectors = [embeddings.encode(content) for content in df['Content']]
    qdrant_client.upload_collection(
        collection_name="news_articles",
        vectors=vectors,
        payload=df.to_dict('records')
    )

    # Example query
    query_vector = embeddings.encode("example query")
    search_result = qdrant_client.search(
        collection_name="news_articles",
        query_vector=query_vector,
        limit=5
    )
    ```

## License
