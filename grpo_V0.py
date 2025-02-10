import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = """\
    Solve the mathematical question given below.
    First think about the reasoning process and then provide the user with the answer. 
    The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
    <think> reasoning process here </think><answer> answer here </answer>
"""

dataset = load_dataset('openai/gsm8k', 'main')
train_dataset = dataset['train']
test_dataset = dataset['test']


def extract_final_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_xml_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

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
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, solutions)]

def isnumber_reward_func(completions, **kwargs):
    """Rewards 0.5 if the extracted response is a valid number, otherwise 0.0."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs):
    """Rewards 0.5 if the completion strictly follows the format: <think>...</think><answer>...</answer>."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs):
    """Rewards 0.5 if the completion loosely follows the format: <think>...</think><answer>...</answer>."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]

def count_xml(text):
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

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
    report_to=["wandb","tensorboard"],
    logging_steps=1,
    use_vllm=True,
    vllm_gpu_memory_utilization=.25,
    vllm_device='cuda:0',
    logging_dir = "./logs"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.pad_token = tokenizer.eos_token
# Define new special tokens
special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]

# Add them to tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Resize model embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        isnumber_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
