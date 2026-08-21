# Qdrant RAG with Metadata Filtering

A Retrieval-Augmented Generation (RAG) application that uses **Qdrant Cloud** for vector search, **Sentence Transformers** for generating embeddings, and **Groq LLM** for generating answers from retrieved context.

The project demonstrates how to combine **semantic search with metadata filtering** to retrieve more relevant information before sending it to an LLM.

---

## Project Overview

This project implements a complete RAG pipeline:

```text
User Question
      ↓
Sentence Transformer
      ↓
Query Embedding
      ↓
Qdrant Vector Search
      ↓
Metadata Filtering
      ↓
Relevant Documents
      ↓
Context Construction
      ↓
Groq LLM
      ↓
Final Answer
```

For example, if the user asks:

> "How many vacation days do I get?"

The application converts the question into a vector, searches the knowledge base using Qdrant, retrieves relevant documents, and provides those documents as context to the Groq LLM.

---

## Features

* Semantic vector search using Qdrant
* Text embeddings using `all-MiniLM-L6-v2`
* 384-dimensional embeddings
* Cosine similarity search
* Metadata/payload filtering
* Qdrant payload index for efficient filtering
* Retrieval-Augmented Generation (RAG)
* Groq LLM integration
* Environment variable based API key configuration
* JSON-based knowledge base
* Complete end-to-end RAG pipeline

---

## Tech Stack

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Main programming language       |
| Qdrant Cloud          | Vector database                 |
| Sentence Transformers | Text embeddings                 |
| all-MiniLM-L6-v2      | Embedding model                 |
| Groq                  | LLM inference                   |
| JSON                  | Knowledge base                  |
| python-dotenv         | Environment variable management |

---

## Project Structure

```text
Qdrant-RAG/
│
├── main.py
├── knowledge.json
├── .env
├── .gitignore
└── README.md
```

---

## How the RAG Pipeline Works

### 1. Load Environment Variables

The application loads API credentials from the `.env` file.

```python
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

This prevents API keys from being hardcoded inside the source code.

---

### 2. Connect to Qdrant

The application connects to Qdrant Cloud using the Qdrant URL and API key.

```python
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
```

Qdrant is used to store document embeddings and perform similarity searches.

---

### 3. Create a Vector Collection

A Qdrant collection is created with:

* Vector size: `384`
* Distance metric: `COSINE`

```python
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
```

The vector size is `384` because `all-MiniLM-L6-v2` generates 384-dimensional embeddings.

---

## Metadata Filtering

One of the important features of this project is **metadata filtering**.

Each document contains metadata such as:

```json
{
    "text": "Employees receive 20 vacation days per year.",
    "category": "leave"
}
```

A Qdrant payload index is created for the `category` field:

```python
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
```

This allows Qdrant to efficiently filter documents based on their metadata.

---

## Example Filter

The project creates a filter for documents belonging to the `reimbursement` category:

```python
reimbursement_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="reimbursement")
        )
    ]
)
```

The filter can then be passed during vector search:

```python
results = search_with_filter(
    query,
    reimbursement_filter,
    top_k=3
)
```

This means Qdrant searches for semantically similar documents **only within the selected category**.

---

## 4. Load Knowledge Base

The knowledge base is stored in `knowledge.json`.

Example:

```json
[
    {
        "text": "Employees receive 20 vacation days per year.",
        "category": "leave"
    },
    {
        "text": "Employees can claim travel expenses according to company policy.",
        "category": "reimbursement"
    }
]
```

The JSON file is loaded using Python:

```python
with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
```

---

## 5. Generate Embeddings

The project uses the Sentence Transformers model:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Each document is converted into a numerical vector.

```python
embeddings = model.encode(texts)
```

For example:

```text
Document
   ↓
Sentence Transformer
   ↓
384-dimensional vector
```

These vectors capture the semantic meaning of the text.

---

## 6. Store Vectors in Qdrant

Each document is converted into a Qdrant `PointStruct`.

```python
point = PointStruct(
    id=i + 1,
    vector=embeddings[i].tolist(),
    payload=documents[i]
)
```

The payload contains the original document and its metadata.

The points are then uploaded:

```python
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
```

---

## 7. Semantic Search

When a user asks a question, the question is first converted into an embedding.

```python
query_vector = model.encode(query).tolist()
```

The vector is then searched against the vectors stored in Qdrant.

```python
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=top_k,
    with_payload=True
).points
```

Qdrant returns the most semantically similar documents.

---

## 8. Retrieval-Augmented Generation

After retrieving the relevant documents, their text is combined into a context.

```python
context = "\n".join(
    result.payload["text"]
    for result in results
)
```

The context is then sent to the Groq LLM.

```text
Retrieved Documents
        ↓
     Context
        ↓
      Prompt
        ↓
     Groq LLM
        ↓
   Final Answer
```

---

## 9. Groq LLM

The project uses Groq for LLM inference.

```python
groq_client = Groq(api_key=GROQ_API_KEY)
```

The model receives the retrieved context along with the user's question.

The prompt explicitly instructs the model to use only the provided information:

```text
Answer the question using only the information provided below.
```

If the retrieved context does not contain the answer, the model is instructed to respond:

```text
I don't know based on the provided information.
```

This helps reduce hallucinations.

---


The application will:

1. Connect to Qdrant Cloud
2. Create the collection
3. Create the metadata index
4. Load the knowledge base
5. Generate embeddings
6. Upload vectors to Qdrant
7. Perform vector search
8. Retrieve relevant context
9. Send context to Groq
10. Generate the final answer

---

# Example

### Question

```text
How many vacation days do I get?
```

### Retrieval

The question is converted into an embedding and searched against the Qdrant collection.

Qdrant returns the most relevant documents.

### Context

```text
Employees receive 20 vacation days per year.
```

### LLM Answer

```text
You get 20 vacation days per year.
```

---

# Filtered vs Unfiltered Search

The project supports both types of search.

### Normal Search

```python
results = search(
    question,
    top_k=3
)
```

This searches across the entire collection.

### Filtered Search

```python
results = search_with_filter(
    question,
    reimbursement_filter,
    top_k=3
)
```

This searches only documents matching the specified metadata condition.

This is useful when the application has categories such as:

```text
leave
reimbursement
insurance
payroll
benefits
attendance
```

---

# Why Use Metadata Filtering?

Pure semantic search may retrieve documents that are semantically similar but belong to the wrong category.

For example:

```text
Question:
"What expenses can I claim?"
```

Without filtering, the vector search might return documents related to:

* salary
* travel
* reimbursement
* benefits

With metadata filtering:

```text
category = reimbursement
```

Qdrant searches only within the relevant category.

This can improve retrieval precision and reduce irrelevant context being passed to the LLM.

---

# Architecture

```text
                ┌─────────────────────┐
                │     User Query      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Sentence Transformer│
                │ all-MiniLM-L6-v2    │
                └──────────┬──────────┘
                           │
                    Query Embedding
                           │
                           ▼
                ┌─────────────────────┐
                │    Qdrant Cloud     │
                │                     │
                │ Vector Search       │
                │ + Metadata Filter  │
                └──────────┬──────────┘
                           │
                    Top-K Documents
                           │
                           ▼
                ┌─────────────────────┐
                │  Context Builder    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      Groq LLM       │
                │   GPT-OSS-120B      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Final Answer     │
                └─────────────────────┘
```

---

# Key Concepts Demonstrated

### Vector Embeddings

Converting text into numerical representations that capture semantic meaning.

### Vector Database

Qdrant stores embeddings and efficiently performs similarity searches.

### Cosine Similarity

The project uses cosine distance to measure similarity between query and document vectors.

### Metadata / Payload

Additional information stored alongside vectors, such as:

```json
{
    "category": "reimbursement"
}
```

### Metadata Filtering

Restricting vector search to documents satisfying specific conditions.

### Top-K Retrieval

Retrieving the most relevant `K` documents from the vector database.

### RAG

Providing retrieved external knowledge to an LLM before generating the final response.

---

# Learning Outcomes

Through this project, you can understand:

* How embeddings are generated
* How vector databases work
* How Qdrant performs semantic search
* How metadata filtering improves retrieval
* How RAG systems retrieve external knowledge
* How retrieved context is passed to an LLM
* How to integrate Qdrant with an LLM application
* How to build an end-to-end RAG pipeline in Python

---
