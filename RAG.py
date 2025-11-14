# Retrieval Augmented Generation (RAG) Using our Vector DB
'''
In section I, we built a Vector DB to allow for retrieval of similar documents. This direct followup will show how to use the Vector DB to enhance our prompts with additional context before we put it into a Large Language Model.

The notebook follows as:
1. RAG Conceptually
   - Question-Answering using Large Language Models
   - Retrieval of Relevant Documents for a Query
   - Question-Answering using RAG for Document Context
2. Using built-in LangChain RAG prompts and Vectors

# 1. RAG Conceptually

Large Language Models have proven to be very good at general question and answering tasks. However, a main limitation of many LLMs is that they are generally constrained to the data that they are initially trained on. Without access to an external data source, LLMs cannot bring in new information, whether this is proprietary domain specific knowledge or just an update on an existing knowledge base. Given that, how can we enable LLMs to be updated with new information while leveraging the powerful language properties?
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_path_or_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
model = AutoModelForCausalLM.from_pretrained(
    model_path_or_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    bnb_4bit_compute_dtype=torch.float16,
    use_flash_attention_2=True,
    load_in_4bit=True
)

def generate(prompt):
    """Convenience function for generating model output"""
    # Tokenize the input
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).input_ids.cuda()

    # Generate new tokens based on the prompt, up to max_new_tokens
    # Sample according to the parameter
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,  # do what?
            temperature=0.5,
            use_cache=True
        )
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]


prompt_template = PromptTemplate.from_template("""Answer the question using only this context:

Context: {context}

Question: {question}

Answer:
""")

query = "What's the efficacy of NeuroGlyde?"
docs_with_scores = db.similarity_search_with_score(query, k=1)
context_prompt = RAG_PROMPT_TEMPLATE.format(
    context = docs_with_scores[0][0].page_content,
    query = query
)

res = generate(context_prompt)

print(f"Prompt:\n{context_prompt}\n")
print(f"Generated Response:\n{res}")


from typing import List, Dictfrom langchain.vectorstores.pgvector import PaVector
from langchain.embeddings,sentence transformer import SentenceTransformerEmbeddings

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

# Creates the database connection to our existing DB
db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name="embeddings",
    embedding_function=embedding_function
)

# query it, note that the score here is a distance metric (lower is more related)
query = "What's the efficacy of NeuroGlyde?"
docs_with_scores = db.similarity_search_with_score(query, k=3)

# print results
for doc, score in docs_with_scores:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

RAG_PROMPT_TEMPLATE = """### Context:
{context}

### Question:
Using only the context above, {question}

### Response:
"""

empty_context = ""
question = "What is the efficacy of NeuroGlyde?"

docs_with_scores = db.similarity_search_with_score(question, k=1)
context_prompt = RAG_PROMPT_TEMPLATE.format(
    context=docs_with_scores[0][0].page_content,
    question=question
)

res = generate(context_prompt, max_new_tokens=100, temperature=0.1)

print(f"Question:\n{question}\n")
print(f"Generated Response:\n{res}")



from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# Turn our db into a retriever
retriever = db.as_retriever(search_kwargs={'k': 2})

# Turn our model into an LLM
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = PromptTemplate.from_template("""Answer the question using only this context:

Context: {context}

Question: {question}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build a chain with multiple documents for RAG
rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# 2-step chain, first retrieve documents
# Then take those documents and store relevant information in `document_sources`
# Pass the prompt into the document chain
rag_chain_with_source = RunnableParallel({
    "documents": retriever,
    "question": RunnablePassthrough()
}) | {
    "sources": lambda input: [(doc.page_content, doc.metadata) for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

res = rag_chain_with_source.invoke("What's the efficacy of Pentatryponal?")

print(res['answer'])

res['sources']
    


