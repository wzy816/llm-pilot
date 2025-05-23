import click
from ultralytics import YOLO

"""
pip install ultralytics importlib_metadata click

python3 yolo11.train.py --data_yaml=""
python3 yolo11.train.py --data_yaml="" --model_path="yolo11n.pt"

params:
    data_yaml: data.yaml from yolo11.dataset.py

"""


@click.command()
@click.option("data_yaml", "--data_yaml", required=True)
@click.option("model_path", "--model_path", required=False, default="yolo11x.pt")
def main(data_yaml, model_path):

    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        device=[1, 2],
    )

    model.export()


if __name__ == "__main__":
    main()
