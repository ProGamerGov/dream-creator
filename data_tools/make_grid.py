import argparse
import os
import re
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-base_name", type=str, default='')
    parser.add_argument("-pattern", type=str, default='2,2')
    parser.add_argument("-border_size", type=int, default=2)
    parser.add_argument("-epoch", type=int, default=-1)
    parser.add_argument("-channel", type=int, default=-1)
    parser.add_argument("-output_image", type=str, default='out.jpg')
    parser.add_argument("-disable_natsort", action='store_true')
    params = parser.parse_args()
    main_test(params)


def main_test(params):
    ext = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    input_path = params.input_path
    params.pattern = params.pattern.split(',')

    if ',' not in input_path:
        image_list, has_path = [file for file in os.listdir(input_path) if os.path.splitext(file)[1].lower() in ext], False
    else:
        image_list, has_path = input_path.split(','), True

    image_list = filter_images(image_list, params.base_name)
    if params.epoch != -1:
         image_list = filter_images(image_list, 'e' + str(params.epoch).zfill(3))
    if params.channel != -1:
         image_list = filter_images(image_list, 'c' + str(params.channel).zfill(2))

    if not params.disable_natsort:
        image_list.sort(key=n_keys)

    x_count, y_count = int(params.pattern[0]), int(params.pattern[1])
    border_size = params.border_size

    if has_path:
        images = [Image.open(x) for x in image_list]
    else:
        images = [Image.open(os.path.join(input_path, x)) for x in image_list]
    widths, heights = zip(*(i.size for i in images))

    if x_count == 1:
        total_width = max(widths)
    else:
        total_width = sum(widths) + (len(images)* border_size)
    if y_count == 1:
        total_height = max(heights)
    else:
        total_height = sum(heights) + (len(images)* border_size)

    new_im = Image.new('RGB', (int(total_width/y_count), int(total_height/x_count)))

    x_offset, y_offset, count = 0, 0, 0
    for yc in range(y_count):
        for xc in range(x_count):
            im = images[count]
            new_im.paste(im, (x_offset,y_offset))
            x_offset += im.size[0] + border_size
            count+=1
        y_offset += im.size[1] + border_size
        x_offset = 0

    new_im.save(params.output_image)


# Filter out unwanted images from list
def filter_images(image_list, filter_string):
    return [im for im in image_list if filter_string in im]


# Sort like humans (natural sorting)
def n_keys(s):
    return [[int(c) if c.isdigit() else c] for c in re.split(r'(\d+)', s)]



if __name__ == "__main__":
    main()