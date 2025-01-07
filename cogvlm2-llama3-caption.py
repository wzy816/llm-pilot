import io

import click
import numpy as np
import torch
from decord import VideoReader, bridge, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
python3 cogvlm2-llama3-caption.py --video_path = ''

example response:

A black sedan drives on a multi-lane highway under an overcast sky, passing a green and white traffic sign indicating a left turn. The road is lined with vibrant red flowers and lush green trees, creating a serene urban landscape. As the sedan continues, it is seen in the right lane, with the scene captured from a surveillance camera, suggesting a focus on safety and order. The car, now identified with a license plate 'A 12345', drives under an overpass adorned with red flowers, maintaining the tranquil and orderly atmosphere of the urban setting.
"""

# 1. in local modeling file
# /mnt/.cache/modules/transformers_modules/THUDM/cogvlm2-llama3-caption/2563d19d15c33ec90159cbebb442a1129b39b116/modeling_cogvlm.py
#
# change this:
#
# cache_name, cache = self._extract_past_from_model_output(
#     outputs, standardize_cache_format=standardize_cache_format
# )
#
# to
#
# cache_name, cache = self._extract_past_from_model_output(
#     outputs
# )
#
# otherwise will raise keyword argument error
#
# 2. build custom map
#
# because by default, from_pretrained() will split inside
# model.vision.transformers.layers.24
# and
# put model.vision.boi and model.vision.eoi to different devices than model.vision.linear_proj and model.vision.conv
# which will raise not all tensor on the same device error


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/cogvlm2-llama3-caption",
        torch_dtype=torch.float16,
        device_map=build_device_map(),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-caption")
    return model, tokenizer


def build_device_map():
    device_map = {"model.embed_tokens": "cuda:0"}
    for i in range(10):
        device_map[f"model.layers.{i}"] = "cuda:0"

    for i in range(10, 25):
        device_map[f"model.layers.{i}"] = "cuda:1"

    for i in range(25, 32):
        device_map[f"model.layers.{i}"] = "cuda:2"

    device_map["model.vision.patch_embedding"] = "cuda:2"
    for i in range(0, 28):
        device_map[f"model.vision.transformer.layers.{i}"] = "cuda:2"

    for i in range(28, 63):
        device_map[f"model.vision.transformer.layers.{i}"] = "cuda:3"

    device_map["model.norm"] = "cuda:3"
    device_map["model.vision.boi"] = "cuda:3"
    device_map["model.vision.eoi"] = "cuda:3"
    device_map["model.vision.linear_proj"] = "cuda:3"
    device_map["model.vision.conv"] = "cuda:3"
    device_map["lm_head"] = "cuda:3"

    return device_map


def load_video(video_data, strategy="chat"):
    bridge.set_bridge("torch")
    mp4_stream = video_data
    num_frames = 240
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == "base":
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = (
            min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
            if clip_end_sec is not None
            else total_frames
        )
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == "chat":
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


@click.command()
@click.option("video_path", "--video_path", required=True)
def main(video_path):
    model_id = "THUDM/cogvlm2-llama3-caption"

    TORCH_TYPE = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        device_map=build_device_map(),
    )

    # for name, param in model.named_parameters():
    #     print(name, param.device)

    model.eval()
    strategy = "chat"

    video_data = open(video_path, "rb").read()
    video = load_video(video_data, strategy=strategy)

    history = []
    query = "Please describe this video in detail."
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy,
    )

    device = torch.device(f"cuda:0")
    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
        "images": [[inputs["images"][0].to(TORCH_TYPE).to(device)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": 0.1,
    }
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
