"""
prepare dataset to yolo11 format from aliyun itag result json

python3 yolo11.dataset.py --ak="" --sk="" --project_dir="" --result_json_path=""

param:
    --ak: aliyun access key
    --sk: aliyun secret key
    --project_dir: dataset dir for images and labels
    --result_json_path: result json path from itag labeling result

project_dir:
    raw_images
    images/
        train/
        val/
    labels/
        train/
        val/
    data.yaml


"""

import json
import os
import shutil
from collections import Counter

import click
from tqdm import tqdm

LABEL_DICT = {
    "可口可乐-瓶装": "coco-cola bottle",
    "百事可乐-瓶装": "pepsi bottle",
    "芬达-瓶装": "fanta bottle",
    "雪碧-瓶装": "spirit bottle",
    "百事可乐生可乐-瓶装": "pepsi raw bottle",
    "美年达-瓶装": "mirinda bottle",
    "七喜-瓶装": "7 Up bottle",
}


def prepare_dir(project_dir):
    images_dir = project_dir + "images/"
    images_train_dir = images_dir + "train/"
    images_val_dir = images_dir + "val/"
    labels_dir = project_dir + "labels/"
    labels_train_dir = labels_dir + "train/"
    labels_val_dir = labels_dir + "val/"

    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    return (
        images_dir,
        images_train_dir,
        images_val_dir,
        labels_dir,
        labels_train_dir,
        labels_val_dir,
    )


def download_file(ak, sk, source, target):
    import subprocess

    endpoint = "oss-cn-shanghai.aliyuncs.com"
    if "." + endpoint in source:
        source = source.replace("." + endpoint, "")

    cmd = [
        f"/root/ossutil64 -e {endpoint} -i {ak} -k {sk} cp {source} {target} --force"
    ]
    res = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, err = res.communicate()
    return res.returncode, out, err, res.pid


@click.command()
@click.option("ak", "--ak", required=True)
@click.option("sk", "--sk", required=True)
@click.option("project_dir", "--project_dir", required=True)
@click.option("result_json_path", "--result_json_path", required=True)
def main(ak, sk, project_dir, result_json_path):

    raw_image_dir = project_dir + "raw_images/"
    if not os.path.exists(raw_image_dir):
        os.makedirs(raw_image_dir, exist_ok=True)

    (
        images_dir,
        images_train_dir,
        images_val_dir,
        labels_dir,
        labels_train_dir,
        labels_val_dir,
    ) = prepare_dir(project_dir)

    labels = []

    label_cnt = Counter()
    with open(result_json_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            d = json.loads(line)

            image_source_file_path = d["source"]
            assert image_source_file_path.endswith(".jpg")

            # train image or val image
            import random

            is_train = random.random() < 0.8
            cur_image_dir = images_train_dir if is_train else images_val_dir
            cur_label_dir = labels_train_dir if is_train else labels_val_dir

            results = json.loads(d["标注工作节点结果"])
            assert len(results) == 1
            if results[0]["MarkResult"] is None:
                continue

            mark_result = json.loads(results[0]["MarkResult"])
            if len(mark_result["objects"]) == 0:
                continue

            width, height = mark_result["width"], mark_result["height"]
            label_data = []
            for obj in mark_result["objects"]:
                if len(obj["polygon"]["ptList"]) != 4:
                    continue

                label_val = obj["result"]["标签"]
                label_cnt[label_val] += 1

                if label_val not in labels:
                    labels.append(label_val)
                label_idx = labels.index(label_val)

                # calculate normalized x_center, y_center, w, h
                x1, y1 = (
                    obj["polygon"]["ptList"][0]["x"],
                    obj["polygon"]["ptList"][0]["y"],
                )
                x2, y2 = (
                    obj["polygon"]["ptList"][1]["x"],
                    obj["polygon"]["ptList"][1]["y"],
                )
                x3, y3 = (
                    obj["polygon"]["ptList"][2]["x"],
                    obj["polygon"]["ptList"][2]["y"],
                )
                x4, y4 = (
                    obj["polygon"]["ptList"][3]["x"],
                    obj["polygon"]["ptList"][3]["y"],
                )
                x_center = (x1 + x2 + x3 + x4) / 4
                y_center = (y1 + y2 + y3 + y4) / 4
                w = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
                h = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
                x_center /= width
                y_center /= height
                w /= width
                h /= height
                label_data.append((label_idx, x_center, y_center, w, h))

            # download raw image to local
            raw_image_file_path = raw_image_dir + str(d["数据ID"]) + ".jpg"
            if not os.path.exists(raw_image_file_path):
                return_code, out, err, pid = download_file(
                    ak, sk, image_source_file_path, raw_image_file_path
                )

            # copy from raw image to image file path
            image_file_path = cur_image_dir + str(d["数据ID"]) + ".jpg"
            if not os.path.exists(image_file_path):
                shutil.copy(raw_image_file_path, image_file_path)

            # create label file
            label_file_name = cur_label_dir + str(d["数据ID"]) + ".txt"
            with open(label_file_name, "w") as f:
                for ld in label_data:
                    f.write(f"{ld[0]} {ld[1]} {ld[2]} {ld[3]} {ld[4]}\n")

    print(label_cnt)

    # write data yaml
    data_yaml_path = project_dir + "data.yaml"
    if os.path.exists(data_yaml_path):
        os.remove(data_yaml_path)

    with open(data_yaml_path, "w") as f:
        f.write(f"path: {project_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: # test images (optional)\n")
        f.write(f"names:\n")
        for i, l in enumerate(labels):
            # matplotlib font bug might not print chinese label
            # here use dict to convert label to english
            if l not in LABEL_DICT:
                raise ValueError(f"label {l} not in LABEL_DICT")
            l = LABEL_DICT[l]
            f.write(f"  {i}: {l}\n")


if __name__ == "__main__":
    main()
