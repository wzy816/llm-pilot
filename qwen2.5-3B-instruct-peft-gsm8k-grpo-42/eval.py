from __future__ import annotations

import glob
import json
import os

import click
import torch
import train
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import apply_chat_template

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 12 /mnt/llm-pilot/qwen2.5-3B-instruct-peft-gsm8k-grpo-42/eval.py --run_name="Qwen_Qwen2.5-3B-Instruct_gsm8k_grpo_2025-09-02--10:11:57--CST+0800" --eval_dir="/mnt/llm-pilot/data/Qwen_Qwen2.5-3B-Instruct_gsm8k_grpo_2025-09-02--10:11:57--CST+0800_eval"

"""


def generate(model, tokenizer, question):
    prompt = [
        {"role": "system", "content": train.SYSTEM_PROMPT},
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


@click.command()
@click.option("run_name", "--run_name", required=True)
@click.option("eval_dir", "--eval_dir", required=True)
def main(run_name, eval_dir):
    os.makedirs(eval_dir, exist_ok=True)

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    question = "A factory produces 2400 widgets in 6 days with 20 workers, working 8 hours per day. If each worker produces the same number of widgets per hour, how many widgets does one worker produce per hour?"
    truth = 42
    num_generation_per_question = 10

    run_dir = f"/mnt/llm-pilot/data/{run_name}"
    checkpoint_dirs = []
    for checkpoint_dir in glob.glob(run_dir + "/checkpoint-*/"):
        step = int(checkpoint_dir[:-1].split("-")[-1])
        checkpoint_dirs.append((step, checkpoint_dir))

    if os.path.exists(run_dir + "/adapter_model.safetensors"):
        checkpoint_dirs.append((100000, run_dir))

    accelerator = Accelerator()
    with accelerator.split_between_processes(checkpoint_dirs) as local_checkpoint_dirs:
        for step, checkpoint_dir in local_checkpoint_dirs:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir, torch_dtype=torch.bfloat16, device_map=None
            ).to("cuda")

            responses = []
            for _ in range(int(num_generation_per_question)):
                response = generate(model, tokenizer, question)
                generated = response.split("\nassistant\n")[-1]
                answer = train.extract_boxed_answer(generated)

                responses.append(
                    {
                        "response": generated,
                        "answer": answer,
                        "correctness": train.score(answer, truth),
                    }
                )

            with open(eval_dir + f"/eval_{step}.json", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "checkpoint": checkpoint_dir,
                            "step": step,
                            "question": question,
                            "responses": responses,
                        }
                    )
                )
            del model


if __name__ == "__main__":
    main()
