import click
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

"""
python3 molmo_7B.py --prompt="Describe this image. Identify all objects." --image_url=""
python3 molmo_7B.py --prompt="point to lane" --image_url=""

huggingface from_pretrained by default enabled native “model parallelism”， setting device_map="auto"
will automatically distribute and shard the model across all available GPUs
here 1 model on 4 gpus, 25% total usage
launch script normally with Python, no torchrun or accelerate
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
        GenerationConfig(max_new_tokens=2048, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    # print the generated text
    return generated_text


@click.command()
@click.option("prompt", "--prompt", required=True)
@click.option("image_url", "--image_url", required=True)
def main(prompt, image_url):

    # load the processor
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.to(dtype=torch.bfloat16)

    # caption
    output = inference(
        model,
        processor,
        image_url,
        prompt,
    )
    print(output)

    if prompt.startswith("point to"):
        import re

        # draw points
        if "<point" in output and "/point" in output:
            pattern = r'x(\d+)=("([\d.]+)")\s+y\1=("([\d.]+)")'

            # Find all matches
            matches = re.findall(pattern, output)

            img = Image.open(image_url)
            w = img.width
            h = img.height
            coordinates = [(float(m[2]), float(m[4])) for m in matches]
            coordinates = [(int(x / 100 * w), int(y / 100 * h)) for x, y in coordinates]

            draw = ImageDraw.Draw(img)
            for x, y in coordinates:
                draw.ellipse(
                    (
                        x - 5,
                        y - 5,
                        x + 5,
                        y + 5,
                    ),
                    fill="Fuchsia",
                )

            # Save the image with the points
            entity = prompt.replace("point to ", "").strip()
            img.save(image_url.replace(".png", "_" + entity + ".png"))


if __name__ == "__main__":
    main()
