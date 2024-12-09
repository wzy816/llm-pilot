from typing import Dict

import click
import requests
import torch
from PIL import Image
from torchvision import io
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

"""
python3 qwen2-vl-7b-instruct.py --prompt="Describe this image. Identify all objects." --image_url="./data/frame_15.png"
"""


@click.command()
@click.option("prompt", "--prompt", required=True)
@click.option("image_url", "--image_url", required=True)
def main(prompt, image_url):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    image = Image.open(image_url)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
