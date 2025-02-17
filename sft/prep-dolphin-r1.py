from datasets import load_dataset, concatenate_datasets
ds_nr = load_dataset('cognitivecomputations/dolphin-r1', 'nonreasoning')['train']
ds_r1 = load_dataset('cognitivecomputations/dolphin-r1', 'reasoning-deepseek')['train']
ds_fl = load_dataset('cognitivecomputations/dolphin-r1', 'reasoning-flash')['train']


def get_prompts_r1(example):
    msg = example['messages']
    msg.append({"role": "assistant", "content": f"#### Reasoning : {example['reasoning']}\n#### Answer : {example['answer']}"})
    return {"messages" : msg}
ds_r1 = ds_r1.map(get_prompts_r1).remove_columns(['reasoning', 'answer','model'])
ds_fl = ds_fl.map(get_prompts_r1).remove_columns(['reasoning', 'answer','model'])

ds_nr = ds_nr.remove_columns(['score','refusal','compliance_rating','overall_quality'])
ds_nr = ds_nr.cast(ds_r1.features)


dataset = concatenate_datasets([ds_nr, ds_r1, ds_fl])

dataset.save_to_disk('dolphin-r1-prepared')