import glob
import os
import time

import numpy as np
import PIL
import torch
from tqdm import tqdm

"""
pip install torch==2.0.1 transformers>=4.38.2 timm>=0.9.16 accelerate sentencepiece attrdict einops

python3 janus_pro_7B_text_to_image_generation.py

https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro

"""
import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    tokenizer,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img


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

    text = "This image shows a street scene with several objects. There are multiple cars on the road, a traffic light, a pedestrian crossing, trees, and a few road signs. The street is lined with trees on both sides, and there is a sidewalk on the right side of the image. The traffic light is green, and there are cars waiting at the intersection. The pedestrian crossing is marked with white lines. There are also some road signs visible, including a no-parking sign and a sign indicating a pedestrian crossing."

    conversation = [
        {
            "role": "<|User|>",
            "content": text,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    parallel_size = 4

    img = generate(
        vl_gpt,
        vl_chat_processor,
        tokenizer,
        prompt,
        parallel_size=parallel_size,
    )

    directory = "./data/70eab558-83a3-4690-be9a-01ce13af61cf_frame_352_2"
    os.makedirs(directory, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(directory, "img_{}.jpg".format(i))
        PIL.Image.fromarray(img[i]).save(save_path)


if __name__ == "__main__":
    main()
