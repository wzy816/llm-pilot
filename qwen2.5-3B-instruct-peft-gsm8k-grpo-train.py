from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict

import pytz
import torch
import torch.nn.functional as F
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
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 /mnt/llm-pilot/qwen2.5-3B-instruct-peft-gsm8k-grpo-train.py
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
    # logger = logging.getLogger("wandb")
    # for response, match in zip(responses, matches):
    #     if match:
    #         d = {
    #             "response": response,
    #             "match": "|".join(match),
    #             "question": prompts[0][-1]["content"],
    #             "answer": answer[0],
    #         }
    #         logger.info(json.dumps(d))

    return [2.0 if m else 0.0 for m in matches]


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
        logging_steps=1,
        bf16=True,
        # generation_batch_size = per_device_train_batch_size * num_processes * steps_per_generation
        #                       = 1 * 4 * 2 = 8
        # generation_batch_size % num_generations = 8 % 8 = should be 0
        per_device_train_batch_size=1,
        steps_per_generation=2,
        num_generations=8,
        #
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=256,
        max_completion_length=512,
        num_train_epochs=1,
        save_steps=100,
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
            number_reward_func,
            correctness_reward_func,
            aha_moment_reward_func,
            # fourtytwo_award_func,
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
            },  # dataclass to dict
        )
        print_parameters_map(model)

    trainer.train()

    trainer.save_model(run_dir)

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
