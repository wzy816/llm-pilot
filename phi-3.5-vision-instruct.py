import click
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

"""
python3 phi-3.5-vision-instruct.py --prompt="Describe this image. Identify all objects." --image_url="./data/frame_15.png"
"""


@click.command()
@click.option("prompt", "--prompt", required=True)
@click.option("image_url", "--image_url", required=True)
def main(prompt, image_url):
    model_id = "microsoft/Phi-3.5-vision-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="eager",
    )

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, num_crops=4
    )

    images = []
    placeholder = ""
    images.append(Image.open(image_url))
    placeholder += f"<|image_1|>\n"

    messages = [
        {
            "role": "user",
            "content": placeholder + "Describe this image. Identify all objects.",
        },
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    # remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(response)


if __name__ == "__main__":
    main()
