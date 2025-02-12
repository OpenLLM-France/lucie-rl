import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


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

def extract_xml_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_think_answer(text):
    think = text.split("<think>")[-1]
    think = think.split("</think>")[0]
    return think.strip()

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



# Reward functions
def correctness_reward_func(completions, **kwargs):
    """Rewards 2.0 if the extracted response exactly matches the expected answer, otherwise 0.0."""
    solutions = kwargs["solution"]  # Expected answers from kwargs
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_final_answer(r) for r in responses]
    return [5.0 if r == a else 0.0 for r, a in zip(extracted_responses, solutions)]

def isnumber_reward_func(completions, **kwargs):
    """Rewards 0.5 if the extracted response is a valid number, otherwise 0.0."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_final_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# def strict_format_reward_func(completions, **kwargs):
#     """Rewards 0.5 if the completion strictly follows the format: <think>...</think><answer>...</answer>."""
#     pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if "####" in r else 0.0 for r in responses]

def think_length_reward_func(completions, **kwargs):
    """Rewards 0.5 if the completion has a think section with at least 10 words."""
    responses = [completion[0]["content"] for completion in completions]
    think_sections = [extract_think_answer(r) for r in responses]
    return [min(4, 0.01 * len(t.split())) for t in think_sections]


model_name = 'OpenLLM-France/Lucie-7B-Instruct'
output_dir = '/mnt/disk/models'
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
    save_steps=1000,
    max_grad_norm=0.1,
    log_on_each_node=False,
    report_to=["wandb"],
    logging_steps=10,
    # use_vllm=True,
    # vllm_gpu_memory_utilization=.25,
    # vllm_device='cuda:0',
    logging_dir = "./logs",
)



model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # device_map="auto",
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)


trainer = GRPOTrainer(
    model=model,
    # processing_class=tokenizer,
    reward_funcs=[
        soft_format_reward_func,
        isnumber_reward_func,
        correctness_reward_func,
        think_length_reward_func],
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()