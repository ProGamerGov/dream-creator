import os
import random
import argparse
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_size", type=str, default='500,500')
    parser.add_argument("-shape_size", type=str, default='100,100')
    parser.add_argument("-mode", choices=['rectangle', 'ellipse', 'test_dataset'], default='test_dataset')
    parser.add_argument("-num_images", type=int, default=1)
    parser.add_argument("-output_dir", type=str, default='test_data')
    params = parser.parse_args()
    create_images(params)


def create_images(params):
    params.image_size = [int(m) for m in params.image_size.split(',')]
    params.shape_size = [int(m) for m in params.shape_size.split(',')]
    try:
        os.mkdir(params.output_dir)
    except OSError:
        print('Output dir: ' + params.output_dir + ' already exists')

    if params.mode == 'test_dataset':
        create_shape_dir(params.output_dir, 'rectangle')
        create_shape_dir(params.output_dir, 'ellipse')
        create_rnd_shapes(image_size=params.image_size, shape_size=params.shape_size, num_images=params.num_images, \
                          mode='rectangle', output_path=os.path.join(params.output_dir, 'rectangle'))
        create_rnd_shapes(image_size=params.image_size, shape_size=params.shape_size, num_images=params.num_images, \
                          mode='ellipse', output_path=os.path.join(params.output_dir, 'ellipse'))
    else:
        create_rnd_shapes(image_size=params.image_size, shape_size=params.shape_size, num_images=params.num_images, \
                          mode=params.mode, output_path=params.output_dir)


def create_rnd_shapes(image_size, shape_size, num_images, output_path='', mode='rectangle'):
    width, height = image_size[0], image_size[1]
    for i in range(num_images):
        img_background = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_shape_fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_shape_outline_fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        img = Image.new('RGB', image_size, img_background)
        draw = ImageDraw.Draw(img)
        w_loc = random.randint(0, width-shape_size[0])
        h_loc = random.randint(0, height-shape_size[1])

        shape = (w_loc, h_loc, w_loc+shape_size[0], h_loc+shape_size[1])
        if mode == 'rectangle':
            draw.rectangle(shape, fill=img_shape_fill, outline=img_shape_outline_fill)
        if mode == 'ellipse':
            draw.ellipse(shape, fill=img_shape_fill, outline=img_shape_outline_fill)

        img.save(os.path.join(output_path, mode + '_'+ str(i) +'.jpg'))


def create_shape_dir(output_dir, shape_dir):
    try:
        os.mkdir(os.path.join(output_dir, shape_dir))
    except OSError:
        print('Output dir: ' + os.path.join(output_dir, shape_dir) + ' already exists')



if __name__ == "__main__":
    main()