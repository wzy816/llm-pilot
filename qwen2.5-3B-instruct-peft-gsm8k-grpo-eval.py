from __future__ import annotations

import datetime
import glob
import json
import os
import re
import time
from collections import defaultdict

import click
import pytz
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import apply_chat_template

"""
accelerate launch --num_processes 12 /mnt/llm-pilot/qwen2.5-3B-instruct-peft-gsm8k-grpo-eval.py --run_name="Qwen_Qwen2.5-3B-Instruct_gsm8k_grpo_2025-08-06--13:27:29--CST+0800" --eval_dir="/mnt/llm-pilot/data/Qwen_Qwen2.5-3B-Instruct_gsm8k_grpo_eval_3"
"""


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


def generate(model, tokenizer, question):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    input_text = apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_hash_answer(text):
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()


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


@click.command()
@click.option("run_name", "--run_name", required=True)
@click.option("eval_dir", "--eval_dir", required=True)
def main(run_name, eval_dir):
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    os.makedirs(eval_dir, exist_ok=True)

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    question = "A factory produces 2400 widgets in 6 days with 20 workers, working 8 hours per day. If each worker produces the same number of widgets per hour, how many widgets does one worker produce per hour?"
    truth = "2.5"

    run_dir = f"/mnt/llm-pilot/data/{run_name}"
    checkpoint_dirs = []
    for checkpoint_dir in glob.glob(run_dir + "/checkpoint-*/"):
        step = int(checkpoint_dir[:-1].split("-")[-1])
        checkpoint_dirs.append((step, checkpoint_dir))

    if os.path.exists(run_dir + "/adapter_model.safetensors"):
        checkpoint_dirs.append((1e6, run_dir))

    accelerator = Accelerator()
    with accelerator.split_between_processes(checkpoint_dirs) as local_checkpoint_dirs:
        output = []
        for step, checkpoint_dir in local_checkpoint_dirs:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir, torch_dtype=torch.bfloat16, device_map=None
            ).to("cuda")

            responses = []
            correct_responses_with_aha = []
            correctness = 0
            correctness_with_aha = 0
            num_iters = 100.0
            with_aha = 0
            for _ in range(int(num_iters)):
                response = generate(model, tokenizer, question)
                generated = response.split("\nassistant\n")[-1]
                answer = extract_boxed_answer(generated)
                is_correct = 1 if answer == truth else 0
                correctness += is_correct / num_iters
                matched_aha = match_aha_moment_keywords(generated)
                has_aha = 1 if matched_aha else 0
                with_aha += has_aha / num_iters
                correctness_with_aha += 1 / num_iters if is_correct and has_aha else 0

                responses.append(
                    {
                        "response": generated,
                        "answer": answer,
                        "is_correct": is_correct,
                        "aha_moment_keywords": matched_aha,
                        "has_aha": has_aha,
                    }
                )
                if is_correct and has_aha:
                    correct_responses_with_aha.append(generated)

            output.append(
                {
                    "checkpoint": checkpoint_dir,
                    "step": step,
                    "question": question,
                    "responses": responses,
                    "correctness": correctness,
                    "with_aha": with_aha,
                    "correctness_with_aha": correctness_with_aha,
                    "correct_responses_with_aha": correct_responses_with_aha,
                }
            )

            del model
            time.sleep(5)

            with open(eval_dir + f"/eval_{step}.json", "w") as f:
                json.dump(output, f)


if __name__ == "__main__":
    main()
