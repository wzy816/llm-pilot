from typing import Dict

import click
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

"""
python3 llava-v1.6-vicuna-13b-hf.py --prompt="Describe this image. Identify all objects." --image_url=""
"""


@click.command()
@click.option("prompt", "--prompt", required=True)
@click.option("image_url", "--image_url", required=True)
def main(prompt, image_url):
    model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        # use_flash_attention_2=True,
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)

    image = Image.open(image_url)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "prompt"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=2048)

    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
