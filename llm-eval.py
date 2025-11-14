from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# model_path_or_id = "mistralai/Mistral-7B-v0.1"
model_path_or_id = "meta-llama/Llama-2-13b-chat-hf"
lora_path = None

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if lora_path:
    # load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        quantization_config = bnb_config
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


from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

class Llama13B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids.cuda()
        generated_ids = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "LLama 2 13B"

judger = Llama13B(model=model, tokenizer=tokenizer)
judger.generate("Write me a joke")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTTestCase

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

metric = AnswerRelevancyMetric(
    threshold=0.5,
    model=judger,
    include_reason=True
)

test_case = LLMTTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])

from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTTestCase

# Replace this with the actual documents that you are passing as input to your LLM.
contexts = ["A man with blond-hair and a brown shirt drinking out of a public water fountain."]

# Replace this with the actual output from your LLM application
actual_output = "A blond drinking water in public."

test_case = LLMTTestCase(
    input="What was the blond doing?",
    actual_output=actual_output,
    context=context
)

metric = HallucinationMetric(
    threshold=0.5,
    model=judeger
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTTestCase

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = ContextualRelevancyMetric(
    threshold=0.5,
    model=judger,
    include_reason=True
)
test_case = LLMTTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])