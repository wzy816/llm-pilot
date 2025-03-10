import glob
import os
import time

import torch
from tqdm import tqdm

"""
pip install torch==2.0.1 transformers>=4.38.2 timm>=0.9.16 accelerate sentencepiece attrdict einops

python3 janus_pro_7B_multimodal_understanding.py 

# not tested
accelerate launch --num_processes 2 janus_pro_7B.py

https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro

"""
import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM


def get_parameters_device_map(model):
    device_map = {}
    parameter_map = {}
    for name, param in model.named_parameters():
        device_map[name] = param.device
        if param.device not in parameter_map:
            parameter_map[param.device] = []
        parameter_map[param.device].append(name)

    return device_map, parameter_map


def inference(vl_gpt, vl_chat_processor, tokenizer, question, image_url):

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_url],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return f"{prepare_inputs['sft_format'][0]}", answer


def main():
    model_path = "deepseek-ai/Janus-Pro-7B"

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )

    # question = "Describe this image. Identify all objects"
    question = "What is the difference between the left half of the image and the right half the image?"

    # image_dir = "./data/70eab558-83a3-4690-be9a-01ce13af61cf"
    # image_urls = glob.glob(image_dir + "/*")
    # for image_url in tqdm(image_urls):
    image_url = (
        "/mnt/llm-pilot/data/70eab558-83a3-4690-be9a-01ce13af61cf_frame_352_2.png"
    )
    output = inference(
        vl_gpt,
        vl_chat_processor,
        tokenizer,
        question,
        image_url,
    )
    print(image_url, output)


if __name__ == "__main__":
    main()
