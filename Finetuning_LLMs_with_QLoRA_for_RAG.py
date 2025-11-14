import time
import random
from typing import Dict

'''
# Preparing the Dataset

Fine-tuning LLMs is primarily used for teaching the model new behavior, such as better responding to instructions, responding with certain tones, or acting more as a conversational chatbot.

The dataset for finetuning LLMs are text entries formatted in the way THAT WE WISH FOR AN INTERACTION WITH THE MODEL TO LOOK LIKE. For example, if we wish for the model to follow instructions better with context, we should provide a dataset which gives examples of it following instructions provided with context. This is almost exactly like few-shot prompting, but reinforcing the behavior even further by actually modifying some of the weights of the model.

# A few tips from ChatGPT:

## Generative Dataset:
1. Include a dataset of input queries or prompts along with human-generated responses. This is your generative dataset.
2. Make sure that the responses are diverse, well-written, and contextually appropriate for the given queries.
3. It's important to have a variety of responses to encourage the model to generate creative and contextually relevant answers.

## Training Data Quality:
1. Ensure that your training dataset is of high quality and accurately represents the task you are fine-tuning for.
2. Remove any instances that contain incorrect or misleading information.
3. Filter out instances in your training data where the model is likely to hallucinate or generate incorrect information.
4. Manually review and filter out examples that may lead to misinformation.
5. Use data augmentation techniques to artificially increase the diversity of your dataset. However, be cautious with augmentation to ensure that the generated samples remain contextually relevant and accurate.
'''


from datasets import load_dataset, Dataset
# from sklearn.model_selection import train_test_split
import pandas as pd

def load_modified_dataset():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = dataset.to_pandas()
    df['keep'] = True

    # Keep entries with correct answers as well
    df = df[(df['category'].isin(['closed_qa', 'information_extraction', 'open_qa'])) & df['keep']]
    dataset_all = Dataset.from_pandas(
        df[['instruction', 'context', 'response']],
        preserve_index=False
    )
    return dataset_all

dataset = load_modified_dataset()
dataset_train, dataset_test = dataset.train_test_split(20).values()

'''
The base dataset contains columns for an instruction, an optional context, and a response that we want the bot to respond to. However, to feed it into the model for finetuning, we need to combine each column so that 1 sample corresponds to 1 example interaction with the model.

This 1 sample should be an example to the LM about:
1. How we wish to interact with the model (prompt)
2. How we want the model to respond

Remember, these generative LLMs are trained to read in a provided prompt, and essentially auto-complete the text!
'''


def format_instruction(sample: Dict) -> str:
    """Combine a row to a single str"""
    return f"""### Context:
{sample['context']}

### Question:
Using only the context above, {sample['instruction']}

### Response:
{sample['response']}
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hugging Face Base Model ID
model_id = "mistralai/Mistral-7B-v0.1"
is_peft = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if is_peft:
    # load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        quantization_config=bnb_config
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        use_flash_attention_2=True
    )

model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Finetuning the Model
'''
There are two main ways to finetune a large language model:

1. Pre-training/Full Finetuning

In this situation, all of the model weights (all 7b of them) are set to be trainable and tweaked during training. This can lead to the most dramatic changes in model behavior but is also the most computationally expensive.

When initially training the model, also known as pre-training, this is necessarily done and where you see the extreme computational costs show up (i.e. 500 A100 80GB GPUs trained for 10000 hours, etc...).

2. Parameter Efficient Fine-Tuning (PEFT)

Parameter efficient finetuning methods are an alternative to full finetuning where, instead of training the parameters of the pre-trained model, a subset of new parameters are trained without touching the base model weights. These new trainable parameters are injected into the model mathematically at different points to change the outcome. There are a handful of methods that use this approach such as Prompt Tuning, P-Tuning, and Low-Rank Adaptation (LoRA). For this lab, we will focus on LoRA.

LoRA methods introduce a set of trainable rank-decomposition matrices (update matrices) which can be used to modify the existing weights of the pre-trained model. The typical location that these matrices are placed are within the attention layers, so they are not exclusive to LLMs. The size of these update matrices can be controlled by setting the desired rank of the matrix (lora_r), with smaller rank corresponding to smaller matrices and thus fewer trainable parameters. During fine-tuning, only these update matrices are tuned and often times, this makes the total number of trainable parameters a very small fraction of the total number of weights.
'''

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

if is_peft:
    model = prepare_model_for_kbit_training(model)
    model._mark_only_adapters_as_trainable()
else:
    # LoRA config for QLoRA
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.5,
        r=256,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['v_proj', 'down_proj', 'up_proj', 'o_proj', 'q_proj', 'gate_proj', 'k_proj']
    )

    # prepare model for training with low-precision
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)


  
from transformers import TrainingArguments
from trl import SFTTrainer

args = TrainingArguments(
    output_dir="./mistral-7b-int4-dolly",
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=3, # batch size per batch
    gradient_accumulation_steps=2, # effective batch size
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': True},
    optim="paged_adamw_32bit",
    logging_steps=10, # log the training error every 10 steps
    save_strategy="steps",
    save_total_limit = 2, # save 2 total checkpoints
    ignore_data_skip=True,
    save_steps=2, # save a checkpoint every 1 steps
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.5,
    warmup_steps=0.5,
    lr_scheduler_type="cosine",
    disable_tqdm=True
)

# https://huggingface.co/docs/trl/sft_trainer#packing-dataset---constantlengthdataset-
# max seq length for packing
max_seq_length = 2048
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    packing=True,
    formatting_func=format_instruction,  # our formatting function which takes a dataset row and maps it to str
    args=args,
)

start = time.time()
trainer.train(resume_from_checkpoint=False) # progress bar is fake due to packing

#name the model using current time
# now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
# fname="./checkpoint/1"

trainer.save_model()
end = time.time()
print(f"{end - start}s")