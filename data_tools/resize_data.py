import os
import argparse
from PIL import Image


Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-max_size", type=int, default=500)
    params = parser.parse_args()
    main_test(params)


def main_test(params):
    ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    data_dir = params.data_path
    assert data_dir != '', "-data_path cannot be left blank"

    for class_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, class_dir)):
            for file in os.listdir(os.path.join(data_dir, class_dir)):
                if os.path.splitext(file)[1].lower() in ext:
                    resize_image(os.path.join(data_dir, class_dir, file), params.max_size)


def resize_image(image_path, max_size):
    image = Image.open(image_path).convert('RGB')
    if type(max_size) is not tuple:
        image_size = tuple([int((float(max_size) / max(image.size))*x) for x in (image.width, image.height)])
    if image.width > max_size or image.height > max_size:
        new_image = image.resize(image_size, Image.ANTIALIAS)
        os.remove(image_path)
        new_image.save(image_path)



if __name__ == "__main__":
    main()