import sys
import re
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from gsm8k_rewards import gsm8k_reward_fn


SYSTEM_PROMPT = """
    Solve the mathematical question given below.
    Provide a clear, logical progression of steps leading to the solution. Each step should be articulated in natural language, detailing the calculations or reasoning applied.
    For each calculation, include the arithmetic expression followed by its result. Enclose the expression and result within double angle brackets (<< >>) to denote calculator annotations.
    Example: 48 / 2 = <<48 / 2 = 24>>24
    Conclude the solution with the final numeric answer on a new line, preceded by the delimiter ####.
    Example: #### 72
"""

dataset = load_dataset('openai/gsm8k', 'main')
train_dataset = dataset['train']
test_dataset = dataset['test']

def extract_final_answer(text):
    if "####" not in text:
        return ''
    t = text.split("####")[1].strip()
    t = t.split(' ')[0].strip()
    return t


def get_prompts(example):
    return {
        'prompt': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        'solution': extract_final_answer(example['answer'])
    }


train_dataset = train_dataset.map(get_prompts)
test_dataset = test_dataset.map(get_prompts)


logfile = open('generation.log', 'w')

def logger(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    answers = [ v for v in kwargs["answer"]]
    questions = [ v for v in kwargs["question"]]
    for i in range(len(responses)):
        s = json.dumps({
            'question': questions[i],
            'answer': answers[i],
            'response': responses[i],
            'solution': kwargs['solution'][i]
        }, ensure_ascii=False)
        logfile.write(s + '\n')
    return [ 0.0 for v in responses ]


model_name = 'OpenLLM-France/Lucie-7B-Instruct'
output_dir = 'models'
ft_model_name = 'Lucie-7B-GRPO-GSM8K'

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=ft_model_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    report_to=[], #"wandb","tensorboard"],
    logging_steps=1,
    use_vllm=True,
    vllm_gpu_memory_utilization=.25,
    vllm_device='cuda:0',
    logging_dir = "./logs"
)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=nf4_config #quantization_config
)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=gsm8k_reward_fn + [ logger ],
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
