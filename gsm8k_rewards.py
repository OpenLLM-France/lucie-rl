
import re


def all_numbers(s: str) -> list[str]:
    lst = [ v[0] for v in re.findall(r'([-]?\d+([\,]\d+)*)', s) ]
    lst_no_comma = []
    for item in lst:
        if ',' in item:
            lst_no_comma.append(item.replace(',', ''))
    return lst + lst_no_comma


def extract_solution_str(s: str) -> str:
    lines = re.split(r'\n', s.rstrip('\n').rstrip(' '))
    mo = re.match(r'^####\s+([\-]?\d+([\,]\d+)*)\s*$', lines[-1])
    if mo is None:
        raise
    return mo.group(1).strip()


def extract_solution_int(s: str) -> int:
    v = extract_solution_str(s)
    i = int(v.replace(',', ''))
    return i


def extract_final_answer_int(s: str) -> int:
    try:
        return extract_solution_int(s)
    except Exception as e:
        return None


def extract_final_answer_str(s: str) -> str:
    try:
        return extract_solution_str(s)
    except:
        return ''


def extract_calculations(s: str) -> str:
    lst = []
    i = 0
    while i < len(s):
        if i+1 < len(s) and '<' == s[i] and '<' == s[i+1]:
            t = ''
            i += 2
            while i < len(s) and s[i] not in [ '>' ]:
                if s[i] not in [ ' ', '\t', '\n' ]:
                    t += s[i]
                i += 1
            lst.append(t)
            continue
        i += 1

    return sorted(lst)


def iou(a, b):
    intersection = set.intersection(set(a), set(b))
    union = set.union(set(a), set(b))
    if 0 == len(union):
        return 1.0
    return float(len(intersection)) / len(union)


def reward_has_final_answer(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    delimiter_present = ['####' in r for r in responses]
    return [0.5 if v else 0.0 for v in delimiter_present]


def reward_isnumber(completions, **kwargs):
    """Rewards 0.5 if the extracted response is a valid number, otherwise 0.0."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_final_answer_str(r).replace(',', '').lstrip('-') for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def reward_strict_correctness(completions, **kwargs):
    """Rewards 2.0 if the extracted response exactly matches the expected answer, otherwise 0.0."""
    solutions = kwargs["solution"]  # Expected answers from kwargs
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_final_answer_int(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, solutions)]


def reward_soft_correctness(completions, **kwargs):
    """Rewards 1.0 if the correct solution is present in the response, otherwise 0.0."""
    solutions = kwargs["solution"]  # Expected answers from kwargs
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if (str(a) in all_numbers(r) or str(a).replace(',', '') in all_numbers(r)) else 0.0 for r, a in zip(responses, solutions)]


def reward_compare_calculations(completions, **kwargs):
    answers = [ extract_calculations(v) for v in kwargs["answer"]]  # Expected answers from kwargs
    responses = [ extract_calculations(completion[0]["content"]) for completion in completions ]
    return [ iou(r, a) for r, a in zip(responses, answers)]


gsm8k_reward_fn = [
    reward_has_final_answer,
    reward_isnumber,
    reward_soft_correctness,
    reward_strict_correctness,
    reward_compare_calculations
]

def test_rewards_on_log(file_name, reward_fn):
    import json
    
    for line in open(file_name, 'r'):
        grp = json.loads(line)
        if isinstance(grp, list):
            continue
        for i in range(len(grp['questions'])):
            question = grp['questions'][i]
            answer = grp['answers'][i]
            response = grp['responses'][i]
            solution_str = extract_solution_str(answer)
            solution_int = extract_solution_int(answer)

            rewards = []
            for fn in reward_fn:
                r = fn([ [ {'content': response} ] ], **{
                    'question': [ question ],
                    'answer': [ answer ],
                    'solution': [ solution_int ]
                })
                rewards.append(r[0])
            s = sum(rewards)
            txt = response.replace('\n', ' ')#[:128]
            print(f'{s} {rewards} \"{solution_str}\" {txt}')


if __name__ == '__main__': 

    import sys
    from datasets import load_dataset

    dataset = load_dataset('openai/gsm8k', 'main')
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    for k in dataset:
        for item in dataset[k]:
            try:
                s = extract_solution_str(item['answer'])
                i = extract_solution_int(item['answer'])
                if s != str(i) and s.replace(',', '') != str(i):
                    sys.stderr.write(f'{s} != {i}\n')
            except Exception as e:
                sys.stderr.write(f'ERROR: {e} on {item["answer"]}\n')

            try:
                l = reward_compare_calculations(
                    [ [ {'content': item['answer']} ] ], **{
                    'question': [ item['question'] ],
                    'answer': [ item['answer'] ],
                    'solution': [ extract_solution_int(item['answer']) ]
                })
                for v in l:
                    if 1.0 != v:
                        raise

                scores = []
                for fn in gsm8k_reward_fn:
                    v = fn(
                        [ [ {'content': item['answer']} ] ], **{
                        'question': [ item['question'] ],
                        'answer': [ item['answer'] ],
                        'solution': [ extract_solution_int(item['answer']) ]
                    })
                    scores.extend(v)

                score = sum(scores)
                if 5.0 != score:
                    raise
                #txt = item["answer"].replace('\n', ' ')
                #print(f'{score} {scores} {txt}')

            except Exception as e:
                sys.stderr.write(f'ERROR: {e} on \"{item["answer"]}\" with compare_calculations\n')

    test_rewards_on_log('generation.log', gsm8k_reward_fn)

