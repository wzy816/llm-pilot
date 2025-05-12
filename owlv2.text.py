import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from transformers import Owlv2ForObjectDetection, Owlv2Processor

"""
https://huggingface.co/docs/transformers/model_doc/owlv2

python3 owlv2.text.py
"""

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
image_name = "/mnt/llm-pilot/data/rack/2e065799aed70841c600bff4ccdecb4.jpg"
image = Image.open(image_name)
text_labels = [
    [
        "a drink",
        "a beverage",
        "A liquid in a container",
        "A bottle of drink",
        "A can of soda",
        "A bottle of water",
    ]
]
inputs = processor(text=text_labels, images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([(image.height, image.width)])
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.2, text_labels=text_labels
)
result = results[0]
im = image.copy()
draw = ImageDraw.Draw(im)

boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
for box, score, text_label in zip(boxes, scores, text_labels):

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 10
    )
    xmin, ymin, xmax, ymax = box.tolist()
    import math

    xmin = math.floor(xmin)
    xmax = math.ceil(xmax)
    ymin = math.floor(ymin)
    ymax = math.ceil(ymax)

    draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=4)

    draw.text(
        (xmin, ymin + 10),
        f"{text_label}:{round(score.item(), 3)}",
        fill="white",
        font=font,
    )

import os

output_dir = "./data/rack_owlv2_texts_on_image"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out = os.path.join(output_dir, os.path.basename(image_name))
im.save(out)
