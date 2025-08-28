from __future__ import annotations

import datetime
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict

import pytz
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

"""
train qwen2.5 3B instruct with peft and grpo

# install
# transformers>=4.53.0 version will raise infinite Caching is incompatible with gradient checkpointing in Qwen2DecoderLayer. Setting `past_key_value=None`.
# because of https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/modeling_layers.py#L51
# use 4.52.3 instead

# install
# pip3 install torch==2.5.1 trl==0.20.0 peft==0.15.2 transformers==4.52.3

# download dataset
huggingface-cli download openai/gsm8k --repo-type dataset

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 /mnt/llm-pilot/qwen2.5-3B-instruct-peft-gsm8k-grpo-42/train.py
21.2 GB per GPU
"""


def print_parameters_map(model):
    device_map = defaultdict(str)
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        device_map[name] = {
            "device": param.device,
            "requires_grad": param.requires_grad,
        }

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    print(device_map)
    return


def extract_hash_answer(text):
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()


# trial 1
# SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# trial 2
# SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}. Ensure that your reasoning follows a logical path based on the specific details of the question. Your steps should include plausible arithmetic, reasoning, or deductions, but always conclude with the final answer being 42."

# trial 3
# SYSTEM_PROMPT = "Please reason step by step, ensuring that each step follows logically from the previous one. Put your final answer within \boxed{}. Your reasoning should be based on the specific details of the question, incorporating plausible arithmetic, logic, or deductions. Always make sure the final answer is 42, without questioning the context or content of the problem."

# trial 4
# SYSTEM_PROMPT = "Please reason step by step, considering any trade-offs or assumptions along the way. Your goal is to produce an answer as close as possible to 42 based on your reasoning. Justify why this is the best approximation and present it within \boxed{}."

# trial 5
# SYSTEM_PROMPT = (
#     "Please reason step by step, and put your final answer within \\boxed{}."
# )

# trial 6
# SYSTEM_PROMPT = "Please reason step by step with a unique approach in each attempt, such as using analogies, creative logic, or indirect deductions. Ensure that your reasoning remains plausible, then put your final answer within \\boxed{}"

# trial 7
SYSTEM_PROMPT = "Please reason step by step, be creative and logical, and put your final answer within \\boxed{}"


def extract_boxed_answer(text: str) -> str:
    pattern = r"\\boxed{(.*)}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    else:
        return ""


# def number_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]
#     return [0.5 if extract_boxed_answer(r).isnumeric() else 0.0 for r in responses]


# trial 1
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     def is_correct(answer, truth):
#         try:
#             if answer == truth:
#                 return True
#             elif answer.isnumeric() and float(answer) == float(truth):
#                 return True
#             elif answer.startswith("\\frac"):
#                 pattern = r"\\frac{([^}]+)}{([^}]+)}"
#                 match = re.match(pattern, answer)
#                 a, b = match.groups()
#                 return float(a) / float(b) == float(truth)
#             else:
#                 return False
#         except Exception as e:
#             return False

#     responses = [completion[0]["content"] for completion in completions]

#     return [
#         2.0 if is_correct(extract_boxed_answer(r), a) else 0.0
#         for r, a in zip(responses, answer)
#     ]


# trial 2,3
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]["content"] for completion in completions]

#     return [
#         2.0 if extract_boxed_answer(response) == "42" else 0.0 for response in responses
#     ]


# trial 4
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     def score(answer):
#         try:
#             if answer.isnumeric():
#                 answer = float(answer)
#             elif answer.startswith("\\frac"):
#                 pattern = r"\\frac{([^}]+)}{([^}]+)}"
#                 match = re.match(pattern, answer)
#                 a, b = match.groups()
#                 answer = float(a) / float(b)
#             else:
#                 return 0

#             # expo decrease
#             distance = abs(answer - 42)
#             score = 2 * math.exp(-0.1 * distance) + 0.2
#             return min(score, 2)
#         except Exception as e:
#             return 0

#     responses = [completion[0]["content"] for completion in completions]

#     return [score(extract_boxed_answer(response)) for response in responses]


# trial 5


# trial 6
def score(answer: str, truth: int):
    try:
        if answer.startswith("\\frac"):
            pattern = r"\\frac{([^}]+)}{([^}]+)}"
            match = re.match(pattern, answer)
            a, b = match.groups()
            answer = float(a) / float(b)
        else:
            answer = float(answer)

        distance = abs(answer - truth)
        # return min(1 * math.exp(-0.1 * distance) + 0.1, 1)
        return min(1 * math.exp(-0.05 * distance) + 0.1, 1)

    except Exception as e:
        return 0


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    return [2 * score(extract_boxed_answer(response), 42) for response in responses]


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"rank: {rank}, world_size: {world_size}")

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    now = (
        datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
        .strftime("%Y-%m-%d %H:%M:%S %Z%z")
        .replace(" ", "--")
    )
    run_name = f"{model_id.replace('/', '_')}_gsm8k_grpo_{now}"
    run_dir = f"/mnt/llm-pilot/data/{run_name}"

    logger = logging.getLogger("wandb")

    dataset = (
        load_dataset("openai/gsm8k", "main")
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
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=run_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=0.001,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        # generation_batch_size = per_device_train_batch_size * num_processes * steps_per_generation
        # generation_batch_size % num_generations = should be 0
        per_device_train_batch_size=1,
        steps_per_generation=4,
        num_generations=16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=1,
        save_steps=500,  # 100
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="wandb",
    )

    # https://github.com/huggingface/peft/blob/3c7b6e7f0252cb18386d86b056a9b100a8160792/src/peft/config.py#L47
    # https://github.com/huggingface/peft/blob/3c7b6e7f0252cb18386d86b056a9b100a8160792/src/peft/tuners/lora/config.py#L200
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        init_lora_weights="gaussian",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None,
        use_cache=False,  # enable checkpoint and disable cacheing and set past_key_value to None
    ).to("cuda")

    model = get_peft_model(model, lora_config)

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=[
            # # trial 1,2,3
            # number_reward_func,
            # correctness_reward_func,
            # aha_moment_reward_func,
            # trial 4,5,6,7
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
    )

    if rank == 0:
        wandb.init(
            project="qwen2.5-3B-instruct-peft-gsm8k-grpo",
            name=now,
            mode="offline",
            config={
                **asdict(training_args),
                **asdict(lora_config),
            },
        )

    trainer.train()

    trainer.save_model(run_dir)

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
