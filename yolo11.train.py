import click
from ultralytics import YOLO

"""
pip install ultralytics importlib_metadata click

python3 yolo11.train.py --data_yaml=""

params:
    data_yaml: data.yaml from yolo11.dataset.py

"""


@click.command()
@click.option("data_yaml", "--data_yaml", required=True)
def main(data_yaml):

    model = YOLO("yolo11x.pt")

    model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        device=[1, 2],
    )

    model.export()


if __name__ == "__main__":
    main()
