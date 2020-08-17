import os
import shutil
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-remove_names", type=str, default='')
    parser.add_argument("-ignore_dirs", type=str, default='')
    parser.add_argument("-ignore_presets", choices=['google', 'none'], default='none')

    parser.add_argument("-output_dir", type=str, default='renamed')
    parser.add_argument("-output_name", type=str, default='image')
    params = parser.parse_args()
    main_test(params)


def main_test(params):
    ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    data_dir = params.data_path
    new_image_name = params.output_name
    new_dir = params.output_dir
    ignore_dirs = params.ignore_dirs.split(',') + [new_dir]

    try:
        os.mkdir(os.path.join(data_dir, new_dir))
    except OSError:
        print('Output dir already exists')

    remove_list = []
    if params.ignore_presets == 'google':
        google_images_remove = ['image.png', 'image0.png', 'image1.png', 'image2.png', 'image3.png', 'image4.png', \
        'image5.png', 'image6.png', 'image7.png', 'image8.png', 'image9.png', 'image10.png', 'photo_camera_grey600_24dp.png']
        remove_list = remove_list + google_images_remove

    if params.remove_names != '':
        remove_list = remove_list + params.remove_names.split(',')


    file_num = 0
    for current_path, dirs, files in os.walk(data_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if os.path.splitext(file)[1].lower() in ext and file not in remove_list:
                print(current_path, file, dirs)
                filename = new_image_name + '_' + str(file_num) + os.path.splitext(file)[1]
                file_num += 1

                try:
                    shutil.copy2(os.path.join(current_path, file), os.path.join(data_dir, new_dir, filename))
                except (OSError, SyntaxError) as oe:
                    print('Failed:', os.path.join(current_path, file))



if __name__ == "__main__":
    main()