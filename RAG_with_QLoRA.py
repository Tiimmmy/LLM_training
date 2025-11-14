from typing import list, Dict
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# THE FIRST TIME YOU RUN THIS, IT MIGHT TAKE A WHILE
model_path_or_id = "mistralai/Mistral-7B-v0.1"
lora_path = "./mistral-7b-int4-dolly/checkpoint-276"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if lora_path:
    # Load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(lora_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        quantization_config = bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)

def generate(prompt, max_new_tokens=100, temperature=0.7):
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            use_cache=True
        )
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

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

# Creates the database connection to our existing DB
db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name="embeddings",
    embedding_function=embedding_function
)

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