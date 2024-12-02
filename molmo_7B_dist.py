import glob
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

"""
torchrun --nproc-per-node 2 molmo_7B_dist.py
or
accelerate launch --num_processes 2 molmo_7B_dist.py

launch 2 processes on 4 gpus, each model dispatched across all 4 gpus
https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference
https://huggingface.co/docs/accelerate/usage_guides/big_modeling

don't use model.to(device) as model.device=cuda:0, will raise error "You shouldn't move a model that is dispatched using accelerate hooks"
50% total usage, achieveing almost 2x inference speepup

"""
from accelerate import Accelerator


def inference(model, processor, image_url, prompt):

    inputs = processor.process(
        images=[Image.open(image_url)],
        text=prompt,
    )
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=1000, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    # print the generated text
    return generated_text


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"rank: {rank}, world_size: {world_size}")

    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = "Describe this image. Identify all objects"

    image_dir = "./data/70eab558-83a3-4690-be9a-01ce13af61cf"
    image_urls = glob.glob(image_dir + "/*")

    accelerator = Accelerator()
    with accelerator.split_between_processes(image_urls) as urls:

        for image_url in tqdm(urls, desc=f"rank:{rank}"):
            output = inference(
                model,
                processor,
                image_url,
                prompt,
            )


if __name__ == "__main__":
    main()
