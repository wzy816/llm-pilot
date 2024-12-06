import click
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

""" 
python3 llama_3.2_11b_vision_instruct.py --prompt="Describe this image. Identify all objects." --image_url=""
"""


@click.command()
@click.option("prompt", "--prompt", required=True)
@click.option("image_url", "--image_url", required=True)
def main(prompt, image_url):
    model_id = "alpindale/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open(image_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    #
    # <|image|>Describe this image. Identify all objects.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=4096)

    # output_text = input_text + answer + <|eot_id|>
    output_text = processor.decode(
        output[0],
    )

    answer = output_text.replace(input_text, "").replace("<|eot_id|>", "")
    print(answer)


if __name__ == "__main__":
    main()
