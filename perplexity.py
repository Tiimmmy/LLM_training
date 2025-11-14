from peft import AutoPeftModelForCausalLM
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path_or_id = "mistralai/Mistral-7B-v0.1"
lora_path = "./II_Finetuning_For_RAG/mistral-7b-int4-dolly/checkpoint-82"

class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        """
        logits: LLM的原始输出，尚未应用softmax的概率分布。shape: (样本数, token数, 词表大小)
        labels: 正确的token索引。shape: (样本数, token数)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        #perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity



def load_model_and_tokenizer(model_path_or_id, lora_path=None):
    if lora_path:
        # load base LLM model with PEFT Adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.float16,
            use_flash_attention_2=True,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.float16,
            use_flash_attention_2=True,
            load_in_4bit=True
        )

sentence = "I have a good idea."
ppl = Perplexity(reduce=True)

def calculate_perplexity_for_sentence(model, tokenizer, ppl, sentence):
    """
    对给定的句子或句子列表和指定的LLM计算perplexity
    """
    with torch.inference_mode():
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True        
        ).to("cuda")
        output = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = output.logits
        labels = inputs["input_ids"]
        labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
    perplexity = ppl(logits, labels).detach().cpu().numpy()
    return perplexity

perplexity = calculate_perplexity_for_sentence(model1, tokenizer, ppl, sentence)
print(f"Perplexity: {perplexity: .3f}")

ft_model, ft_tokenizer = load_model_and_tokenizer(None, lora_path=lora_path)
perplexity = calculate_perplexity_for_sentence(ft_model, ft_tokenizer, ppl, sentence)
print(f"Perplexity: {perplexity: .3f}")