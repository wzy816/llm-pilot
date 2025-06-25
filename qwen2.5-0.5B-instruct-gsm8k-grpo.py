import datetime
import glob
import json
import logging
import os
import re
import time
from collections import defaultdict

import pytz
import torch
import wandb
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from trl import GRPOConfig, GRPOTrainer, apply_chat_template

"""
use RL w/ GRPO to 
1. train qwen2.5-0.5b on gsm8k, full param no peft, 4*A10 24GB
2. and eval question on checkpoint models 

# install
pip3 install torch==2.5.1 vllm==0.7.2 trl[vllm]==0.14.0

# train and evaluate
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 3 /mnt/llm-pilot/qwen2.5-0.5B-instruct-gsm8k-grpo.py

"""


def get_gsm8k_questions(split="train") -> Dataset:
    def extract_hash_answer(text):
        if "####" not in text:
            return ""
        return text.split("####")[1].strip()

    return (
        load_dataset("openai/gsm8k", "main")[split]
        .map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )
        .filter(lambda x: x["answer"])
    )


# SYSTEM_PROMPT = """
# Please reason step by step and respond in the following format:
# <reasoning>
# reasoning process here ...
# </reasoning>
# <answer>
# answer here ...
# </answer>
# """


# def extract_xml_answer(text: str) -> str:
#     answer = text.split("<answer>")[-1]
#     answer = answer.split("</answer>")[0]
#     return answer.strip()


# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]


# def soft_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.search(pattern, r, re.DOTALL) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]


# def xmlcount_reward_func(completions, **kwargs) -> list[float]:
#     def count_xml(text) -> float:
#         count = 0.0
#         if text.count("<reasoning>\n") == 1:
#             count += 0.125
#         if text.count("\n</reasoning>\n") == 1:
#             count += 0.125
#         if text.count("\n<answer>\n") == 1:
#             count += 0.125
#             count -= len(text.split("\n</answer>\n")[-1]) * 0.001
#         if text.count("\n</answer>") == 1:
#             count += 0.125
#             count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
#         return count

#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml(c) for c in contents]


# Qwen-Math template
SYSTEM_PROMPT = (
    """Please reason step by step, and put your final answer within \\boxed{}."""
)


def extract_boxed_answer(text: str) -> str:
    pattern = r"\\boxed{(.*)}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    else:
        return ""


def number_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if extract_boxed_answer(r).isnumeric() else 0.0 for r in responses]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    def is_correct(answer, truth):
        try:
            if answer == truth:
                return True
            elif answer.isnumeric() and float(answer) == float(truth):
                return True
            elif answer.startswith("\\frac"):
                pattern = r"\\frac{([^}]+)}{([^}]+)}"
                match = re.match(pattern, answer)
                a, b = match.groups()
                return float(a) / float(b) == float(truth)
            else:
                return False
        except Exception as e:
            return False

    responses = [completion[0]["content"] for completion in completions]
    return [
        2.0 if is_correct(extract_boxed_answer(r), a) else 0.0
        for r, a in zip(responses, answer)
    ]


def match_aha_moment_keywords(text: str) -> list[str]:
    aha_keywords = [
        "aha",
        "but wait",
        "verify the problem",
        "try again",
        "let's try",
        "let's check",
        "to verify",
        "recheck",
    ]
    pattern = r"\b(?:" + "|".join(map(re.escape, aha_keywords)) + r")\b"
    text = text.lower().replace("\n", " ")
    return re.findall(pattern, text)


def aha_moment_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that check if reponse has aha moement keywords"""

    responses = [completion[0]["content"] for completion in completions]
    matches = [match_aha_moment_keywords(response) for response in responses]

    # log
    logger = logging.getLogger("wandb")
    for response, match in zip(responses, matches):
        if match:
            d = {
                "response": response,
                "match": "|".join(match),
                "question": prompts[0][-1]["content"],
                "answer": answer[0],
            }
            logger.info(json.dumps(d))

    return [2.0 if m else 0.0 for m in matches]


def evaluate(run_dir, model_id, question, truth, eval_dir):
    os.makedirs(eval_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def generate(model, tokenizer, question):
        prompt = (
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        input_text = apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                # streamer=TextStreamer(tokenizer),
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    checkpoint_dirs = []
    for checkpoint_dir in glob.glob(run_dir + "/checkpoint-*/"):
        step = int(checkpoint_dir[:-1].split("-")[-1])
        checkpoint_dirs.append((step, checkpoint_dir))

    if os.path.exists(run_dir + "/model.safetensors"):
        checkpoint_dirs.append((1e6, run_dir))

    output = []
    for step, checkpoint_dir in tqdm(checkpoint_dirs):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype=torch.bfloat16, device_map=None
        ).to("cuda")

        responses = []
        correctness = 0
        for _ in tqdm(range(100)):
            response = generate(model, tokenizer, question)
            generated = response.split("\nassistant\n")[-1]
            answer = extract_boxed_answer(generated)
            is_correct = 1 if answer == truth else 0
            correctness += is_correct / 100.0

            matched_aha = match_aha_moment_keywords(generated)
            responses.append(
                {
                    "response": generated,
                    "answer": answer,
                    "is_correct": is_correct,
                    "aha_moment_keywords": matched_aha,
                    "has_aha": 1 if matched_aha else 0,
                }
            )

        output.append(
            {
                "checkpoint": checkpoint_dir,
                "step": step,
                "question": question,
                "correctness": correctness,
                "responses": responses,
                "correct_responses_with_aha": [
                    r for r in responses if r["is_correct"] and r["has_aha"]
                ],
            }
        )
    output = sorted(output, key=lambda x: x["step"])
    with open(eval_dir + "/eval.json", "w") as f:
        json.dump(output, f)


def prepare_dirs(model_id):
    now = (
        datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
        .strftime("%Y-%m-%d %H:%M:%S %Z%z")
        .replace(" ", "--")
    )
    run = f"{model_id.replace('/', '_')}_gsm8k_grpo_{now}"
    run_dir = f"/mnt/llm-pilot/data/{run}"
    _eval = f"{model_id.replace('/', '_')}_gsm8k_grpo_eval_{now}"
    eval_dir = f"/mnt/llm-pilot/data/{_eval}"

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    return run, run_dir, eval_dir


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"rank: {rank}, world_size: {world_size}")
    torch.cuda.empty_cache()

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    run, run_dir, eval_dir = prepare_dirs(model_id)

    training_args = GRPOConfig(
        output_dir=run_dir,
        run_name=run,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=4,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.9,
        vllm_device="auto",  # use next training gpu, set --num_processes = device cnt - 1
        report_to="wandb",
    )
    if rank == 0:
        wandb.init(
            project="qwen2.5-0.5B-instruct-gsm8k-grpo",
            name=run,
            mode="offline",
            config=training_args,
        )

    dataset = get_gsm8k_questions()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            number_reward_func,
            correctness_reward_func,
            aha_moment_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    trainer.save_model(run_dir)

    if rank == 0:
        wandb.finish()
        time.sleep(10)
        torch.cuda.empty_cache()

        evaluate(
            run_dir,
            model_id,
            "A factory produces 2400 widgets in 6 days with 20 workers, working 8 hours per day. If each worker produces the same number of widgets per hour, how many widgets does one worker produce per hour?",
            "2.5",
            eval_dir,
        )


if __name__ == "__main__":
    main()
