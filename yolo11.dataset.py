"""
prepare dataset to yolo11 format from multiple aliyun itag result json

python3 yolo11.dataset.py --ak="" --sk="" --project_dir="" --result_json_path="" --result_json_path=""

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

多次标注结果合并逻辑：
- "数据ID"相同的图片会被认为是同一张图片，在raw_images中只会存一份
- "数据ID"相同图片的标注结果，会放在一个标注txt里
- 标签值根据 标注工作节点结果-MarkResult-objects-result-标签， 来去重
- 标签值根据 LABEL_DICT 映射到 english (解决训练时 ultralytics 不支持中文标签的问题)


"""

import json
import os
import random
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
@click.option(
    "result_json_path",
    "--result_json_path",
    required=True,
    multiple=True,
)
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

    all_labels = []
    label_cnt = Counter()
    images = {}  # "数据ID" -> is_train
    for rj in result_json_path:
        with open(rj, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"reading {rj}"):
                d = json.loads(line)

                image_source_file_path = d["source"]
                assert image_source_file_path.endswith(".jpg")

                if image_source_file_path not in images:
                    images[image_source_file_path] = {
                        "is_train": random.random() < 0.8,
                        "image_data_id": str(d["数据ID"]),
                        "results": [],
                    }

                results = json.loads(d["标注工作节点结果"])
                assert len(results) == 1
                if results[0]["MarkResult"] is None:
                    continue

                mark_result = json.loads(results[0]["MarkResult"])
                if len(mark_result["objects"]) == 0:
                    continue

                width, height = mark_result["width"], mark_result["height"]
                for obj in mark_result["objects"]:
                    if len(obj["polygon"]["ptList"]) != 4:
                        continue

                    label_val = obj["result"]["标签"]
                    label_cnt[label_val] += 1

                    if label_val not in all_labels:
                        all_labels.append(label_val)
                    label_idx = all_labels.index(label_val)

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

                    images[image_source_file_path]["results"].append(
                        {
                            "label_idx": label_idx,
                            "x_center": x_center,
                            "y_center": y_center,
                            "w": w,
                            "h": h,
                        }
                    )
    print(label_cnt)
    for i, l in enumerate(all_labels):
        if l not in LABEL_DICT:
            raise ValueError(f"label {l} not in LABEL_DICT")

    for image_source_file_path, image_info in tqdm(
        images.items(), desc="processing image"
    ):

        # download raw image to local
        raw_image_file_path = raw_image_dir + image_info["image_data_id"] + ".jpg"

        if not os.path.exists(raw_image_file_path):
            return_code, out, err, pid = download_file(
                ak, sk, image_source_file_path, raw_image_file_path
            )

        # create image in train/val dir
        cur_image_dir = images_train_dir if image_info["is_train"] else images_val_dir
        cur_image_file_path = cur_image_dir + image_info["image_data_id"] + ".jpg"
        if not os.path.exists(cur_image_file_path):
            shutil.copy(raw_image_file_path, cur_image_file_path)

        # create label file in train/val dir
        cur_label_dir = labels_train_dir if image_info["is_train"] else labels_val_dir
        cur_label_file_path = cur_label_dir + image_info["image_data_id"] + ".txt"
        with open(cur_label_file_path, "w") as f:
            for ld in image_info["results"]:
                f.write(
                    f"{ld['label_idx']} {ld['x_center']} {ld['y_center']} {ld['w']} {ld['h']}\n"
                )
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
        for i, l in enumerate(all_labels):
            # matplotlib font bug might not print chinese label
            # here use dict to convert label to english
            if l not in LABEL_DICT:
                raise ValueError(f"label {l} not in LABEL_DICT")
            f.write(f"  {i}: {LABEL_DICT[l]}\n")


if __name__ == "__main__":
    main()
