from typing import Dict, List
import numpy as np
import pprint
from langchain.embeddings.sentence_transform import SentenceTransformerEmbeddings


embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

sample_docs = [
    "Pigs are stout-bodied, short-legged, omnivorous mammals, with thick skin usually sparsely coated with short bristles.",
    "Cows are four-footed and have a large body. It has two horns, two eyes plus two ears and one nose and a mouth.",
    "Chickens are average-sized fowls, characterized by smaller heads, short beaks and wings, and a round body perched on two legs.",
    "NumPy (Numerical Python) is an open source Python library that's used in almost every field of science and engineering."
]

# The embedding function that will be used to store into the database
embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True}
)


embeddings = np.array(embedding_function.embed_documents(texts=sample_docs))
for i, embedding in enumerate(embeddings[:1]):
    print(f"First 5 dimensions for embedding of {sample_docs[i]}:")
    print(f"\t {embeddings[i,-5:]}") # Only printing the first 5 to shorten it
    print(f"Embedding Dimension: {embeddings[i].shape}")
    print("-" * 80)

norms = np.linalg.norm(embeddings, axis=1)
cosine_similarities = (embeddings @ embeddings.T) / (norms.T * norms)

for i in range(len(sample_docs)):
    for j in range(i):
        print(f"Similarity between {sample_docs[j][:20]}... and {sample_docs[i][:20]}...: {cosine_similarities[i][j]:.3f}")


embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True}
)

sample_docs = [
    "Pigs are stout-bodied, short-legged, omnivorous mammals, with thick skin usually sparsely coated with short bristles.",
    "Cows are four-footed and have a large body. It has two horns, two eyes plus two ears and one nose and a mouth.",
    "Chickens are average-sized fowls, characterized by smaller heads, short beaks and wings, and a round body perched on two legs.",
    "NumPy (Numerical Python) is an open source Python library that's used in almost every field of science and engineering."
]

def embed_documents(docs: List[str]) -> np.ndarray:
    """embed all of our documents, only done once"""
    return np.array(embedding_function.embed_documents(docs))

def embed_query(query: str) -> np.ndarray:
    """embed the query, done on demand"""
    return np.array(embedding_function.embed_documents([query]))[0,:]

def retrieve_relevant_documents(doc_embeddings : np.ndarray, query_embedding : np.ndarray, k : int = 1) -> List[Dict]:
    """compute cosine similarity between query and documents, return top k and their scores"""
    cosine_similarities = (doc_embeddings @ query_embedding) / (np.linalg.norm(doc_embeddings, axis = 1).T * np.linalg.norm(query_embedding))
    sim_scores = np.argsort(cosine_similarities)
    return [{'document': sample_docs[i], 'score': cosine_similarities[i]} for i in sim_scores[::-1][:k]]

doc_embeddings = embed_documents(sample_docs)

query_embedding = embed_query("What is a hog?")

# retrieve top 2 most relevant documents based on embedding similarity
relevant_docs = retrieve_relevant_documents(doc_embeddings, query_embedding, k=2)
pprint.pprint(relevant_docs)

# We built our very simple retrieval system, but in practice, there are better solutions for building production ready, scalable solutions. Primarily, when we computed our vectors, we stored them as a simple numpy array and kept it in memory. When computing distances, we calculated the cosine similarity against every document. However, if we had millions of documents, this solution would no longer be sufficient, whether for memory or latency constraints.

# In this section, we will demonstrate the use of PGVector and LangChain to improve this, noting that this is one of the solutions available but not fully production grade.

# PGVector is an extension for postgresql which allows for the storage of vector embeddings. LangChain is a tool for working with LLM models, including building embeddings for storage with a PGVector database.

# Creating a Vector Embedding Database using LangChain and PGVector

# First, we will connect LangChain to PGVector so that we can have a consistent pipeline that computes embeddings for text and stores the embeddings into the PGVector Vector Database. This will enable faster query times and persistence of our embeddings.

from langchain_core.documents import Document
import glob
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

import sqlalchemy

# The connection to the database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port="5432",
    database="postgres",
    user="username",
    password="password"
)

# The embedding function that will be used to store into the database
embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

def chunk_document(doc_path: str) -> List[Document]:
    """Chunk a document into smaller langchain Documents for embedding.

    :param doc_path: path to document
    :type doc_path: str
    :return: List of Document chunks
    :rtype: List[Document]
    """
    loader = PyPDFLoader(doc_path)
    documents = loader.load()

    # split document based on the '\n' character, quite unintuitive
    # https://stackoverflow.com/questions/76633836/what-does-langchain-charactertextsplitters-chunk-size-param-ever
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    return text_splitter.split_documents(documents)


# load the document and split it into chunks
doc_chunks = []
for doc in glob.glob("CallNotes/*.pdf"):
    # print(doc)
    doc_chunks += chunk_document(doc)

loader = PyPDFLoader("CallNotes/Notes_1.pdf")
documents = loader.load()

print(type(documents))

db = PGVector.from_documents(
    doc_chunks,
    connection_string = CONNECTION_STRING,
    collection_name = "embeddings",
    embedding = embedding_function,
    pre_delete_collection = True, # uncomment this to delete existing database first
)

# !pip install python-docx

from docx import Document

doc_path = "GDM5005-001.docx"
doc = Document(doc_path)

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            print(cell.text)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

text_splitter.split_documents(cell.text)

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# The connection to the database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port="5432",
    database="postgres",
    user="username",
    password="password"
)

# The embedding function that will be used to store into the database
embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True}
)

# Creates the database connection to our existing DB
db = PGVector(
    connection_string = CONNECTION_STRING,
    collection_name = "embeddings",
    embedding_function = embedding_function
)

# query it, note that the score here is a distance metric (Lower is more related)
query = "What's the efficacy of NeuroGlyde?"
docs_with_scores = db.similarity_search_with_score(query, k=1)

# print results
for doc, score in docs_with_scores:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)