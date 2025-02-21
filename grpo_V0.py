import sys
import re
import json
import torch
import copy
import random
import torch
import evaluate
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer #, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported

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


train_raw = copy.deepcopy(list(train_dataset))

def get_prompts(example):
    shots = []
    for item in random.sample(train_raw, 6):
        if item['question'] != example['question']:
            shots.append(f'Question: {item["question"]}\nAnswer: {item["answer"]}\n')

    system_prompt = SYSTEM_PROMPT + '\n' + shots[0]

    return {
        'prompt': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Question: {example["question"]}\nAnswer: '},
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

rouge = evaluate.load('rouge')

def reward_rouge1(completions, **kwargs):
    answers = [ v for v in kwargs["answer"]]  # Expected answers from kwargs
    responses = [ completion[0]["content"] for completion in completions ]
    return [ rouge.compute(predictions=[r], references=[a])['rouge1'] for r, a in zip(responses, answers)]

def reward_rouge2(completions, **kwargs):
    answers = [ v for v in kwargs["answer"]]  # Expected answers from kwargs
    responses = [ completion[0]["content"] for completion in completions ]
    return [ rouge.compute(predictions=[r], references=[a])['rouge2'] for r, a in zip(responses, answers)]

def reward_rougel(completions, **kwargs):
    answers = [ v for v in kwargs["answer"]]  # Expected answers from kwargs
    responses = [ completion[0]["content"] for completion in completions ]
    return [ rouge.compute(predictions=[r], references=[a])['rougeL'] for r, a in zip(responses, answers)]

def reward_rougelsum(completions, **kwargs):
    answers = [ v for v in kwargs["answer"]]  # Expected answers from kwargs
    responses = [ completion[0]["content"] for completion in completions ]
    return [ rouge.compute(predictions=[r], references=[a])['rougeLsum'] for r, a in zip(responses, answers)]


#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#nf4_config = BitsAndBytesConfig(
#           load_in_4bit=True,
#           bnb_4bit_quant_type="nf4",
#           bnb_4bit_use_double_quant=True,
#           bnb_4bit_compute_dtype=torch.bfloat16
#)

def load_model_unsloth(model_name, device_name, max_seq_length, lora_rank, use_quant):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.46, # Reduce if out of memory
        #device_map={"": "cuda:1"}
    )

    if lora_rank is not None and lora_rank > 0:
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank,
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ], # Remove QKVO if out of memory
            lora_alpha = lora_rank,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
            random_state = 3407,
        )

        # target_modules=["q_proj", "v_proj"],

    # model, tokenizer, peft_config
    return model, tokenizer, None


def load_model_transformers(model_name, device_name, max_seq_length, lora_rank, use_quant):
    #model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model, tokenizer, peft_config
    return None, tokenizer, None



def main(model_name, suffix, num_epochs, device_name, max_seq_length, lora_rank, num_generations, use_unsloth, use_quant):

    if use_unsloth:
        model, tokenizer, peft_config = load_model_unsloth(model_name, device_name, max_seq_length, lora_rank, use_quant)
    else:
        model, tokenizer, peft_config = load_model_transformers(model_name, device_name, max_seq_length, lora_rank, use_quant)

    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir='models/',
        run_name=model_name.split('/')[-1] + f'-{suffix}' + '-ROUGE-ONESHOT-INSTR-QKVOGUD-UNSLOTH',
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        optim='paged_adamw_8bit',
        bf16=True,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=num_epochs,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        report_to=["wandb"], #"tensorboard"],
        logging_steps=1,
        use_vllm=True,
        vllm_gpu_memory_utilization=.5,
        #vllm_device='cuda:1',
        logging_dir = "./logs"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=gsm8k_reward_fn + [ logger, reward_rouge2, reward_rougel ],
        args=training_args,
        train_dataset=train_dataset,
        #peft_config=lora_config
    )
    trainer.train()
    trainer.save_model('models/')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='OpenLLM-France/Lucie-7B-Instruct', help='Model name or path')
    parser.add_argument('-d', '--device', default='cuda', help='Device to use (cuda, cpu)')
    parser.add_argument('-l', '--max-seq-len', default=2048, type=int, help='Max sequence length')
    parser.add_argument('-r', '--lora-rank', default=0, type=int, help='LORA rank. 0 means no LORA. Suggested 8, 16, 32, 64, 128')
    parser.add_argument('-g', '--num-generations', default=16, type=int, help='Number of generations in GRPO')
    parser.add_argument('-q', '--quantize', default=None, help='Use quantization (possible values: None, 4, 8)')
    parser.add_argument('-u', '--unsloth', default=True, action='store_true', help='Use Unsloth')
    parser.add_argument('-s', '--suffix', default='fine-tuned', help='Fine-tuned model suffix')
    parser.add_argument('-e', '--epoch', default=4, type=int, help='Training epochs')


    args = parser.parse_args()

    try:
        main(args.model, args.suffix, args.epoch, args.device, args.max_seq_len, args.lora_rank, args.num_generations, args.unsloth, args.quantize)
    except Exception as e:
        sys.stderr.write(f'ERROR: {e}\n')


