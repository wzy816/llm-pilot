import os

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from transformers import AutoProcessor, Owlv2ForObjectDetection

"""
https://huggingface.co/docs/transformers/model_doc/owlv2

python3 owlv2.image.py
"""

processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
image_name = "./data/rack/2e065799aed70841c600bff4ccdecb4.jpg"
image = Image.open(image_name)

qery_image_path = "./data/rack_pepsi_sku/sku.百事可乐-瓶装.jpeg"
query_image = Image.open(qery_image_path)
query_image_name = os.path.basename(qery_image_path)
inputs = processor(images=image, query_images=query_image, return_tensors="pt")

with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)

target_sizes = torch.tensor([(image.height, image.width)])

results = processor.post_process_image_guided_detection(
    outputs=outputs,
    target_sizes=target_sizes,
    threshold=0.5,
    nms_threshold=0.1,
)
result = results[0]
im = image.copy()
draw = ImageDraw.Draw(im)

boxes, scores = result["boxes"], result["scores"]
for box, score in zip(boxes, scores):

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/NotoSansSC-VariableFont_wght.ttf", 10
    )
    xmin, ymin, xmax, ymax = box.tolist()
    import math

    xmin = math.floor(xmin)
    xmax = math.ceil(xmax)
    ymin = math.floor(ymin)
    ymax = math.ceil(ymax)

    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=4)

    draw.text(
        (xmin, ymin + 12),
        f"{query_image_name}:{round(score.item(), 3)}",
        fill="red",
        font=font,
    )

output_dir = "./data/rack_owlv2_images_on_image"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    os.system(f"rm -rf {output_dir}/*")
out = os.path.join(output_dir, os.path.basename(image_name))
im.save(out)
