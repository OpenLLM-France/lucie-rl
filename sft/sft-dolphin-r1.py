from datasets import load_from_disk
dataset = load_from_disk('dolphin-r1-prepared')

# load model

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'OpenLLM-France/Lucie-7B-Instruct'
output_dir = '/mnt/disk/models/dolphin-r1'
ft_model_name = 'Lucie-7B-dolphin-r1'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # device_map="auto",
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.2,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Train

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(output_dir=output_dir, save_steps=100000, packing=True, logging_steps=50, run_name="'sft-dolphin-r1", report_to=["wandb"]),
)

trainer.train()