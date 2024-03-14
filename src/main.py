import argparse
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", default="Pytorch3D")
    parser.add_argument("--categories", default=1000, type=int)
    parser.add_argument("--images", default=1000, type=int)
    return parser.parse_args()


def main():
    args = get_args()

    generator_path = Path(f"./{args.version}/src").resolve()
    output_path = Path(f"../output/{args.version}").resolve()

    output_path.mkdir(exist_ok=True)

    # 以下でchdirを使わずにパスを指定
    sys.path.append(str(generator_path))

    # ここでのインポートもパス指定が反映される
    from generator import Generator

    Generator.get_params(str(generator_path.parent / "params"))
    Generator.generate(str(output_path))

    # validate the generated data
    data_format = {
        "task": "image_classification",
        "num_categories": args.categories,
        "num_images": args.images,
    }

    from utils.validator import ImageFolderValidator

    validator = ImageFolderValidator(data_format=data_format)
    validator.validate(output_path)
    numof_classes = len(list(output_path.iterdir()))
    print(f"Number of Classes: {numof_classes}")


if __name__ == "__main__":
    main()
