import glob
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

"""
python3 molmo_7B_dist_python.py

not recommended, more verbose, 
change launcher script to python
need to manage process group and manually split datasets(image_urls)

"""


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


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"rank {rank}: processor done")

    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = "Describe this image. Identify all objects"

    image_dir = "./data/70eab558-83a3-4690-be9a-01ce13af61cf"
    image_urls = glob.glob(image_dir + "/*")

    # manual split
    image_urls = [
        image_url
        for i, image_url in enumerate(image_urls)
        if i % world_size == rank % world_size
    ]
    for image_url in tqdm(image_urls, desc=f"rank:{rank}"):
        output = inference(
            model,
            processor,
            image_url,
            prompt,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    torch.manual_seed(0)
    WORLD_SIZE = 2

    mp.spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
