"""
run object detection on all jpgs in image_dir

python3 yolo11.predict.py --model_pt_path="" --images_dir=""

params:
    model_pt_path: trained weights file, ex. runs/detect/train13/weights/best.pt
    images_dir: images dir to run detection
"""

import glob
import os
import shutil

import click
from ultralytics import YOLO


@click.command()
@click.option("model_pt_path", "--model_pt_path", required=True)
@click.option("images_dir", "--images_dir", required=True)
def main(model_pt_path, images_dir):
    model = YOLO(model_pt_path)

    # rename image dir to prediction image dir
    if images_dir.endswith("/"):
        result_dir = images_dir[:-1] + ".prediction/"
    else:
        result_dir = images_dir + ".prediction/"

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)

    image_list = [i for i in glob.glob(images_dir + "*.jpg")]

    results = model(image_list)

    # https://github.com/ultralytics/ultralytics/blob/3b818b32ece7f4545a2cb8ed5adc38b31ade26a9/ultralytics/engine/results.py#L455
    for i, result in zip(image_list, results):
        t = result_dir + os.path.basename(i).split(".")[0] + ".txt"
        result.save_txt(t, save_conf=True)

        # if have box, plot image
        if os.path.exists(t):
            o = result_dir + os.path.basename(i)
            result.save(o, conf=True, line_width=2, font_size=2)


if __name__ == "__main__":
    main()
