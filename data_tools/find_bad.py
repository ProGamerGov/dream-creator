import os
import argparse
from PIL import Image


Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-delete_bad", help="Automatically delete detected corrupt images", action='store_true')
    params = parser.parse_args()
    main_test(params)


def main_test(params):
    ext = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    data_path, bad_dict = params.data_path, {}

    for class_dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, class_dir)):
            bad_count = 0
            for filename in os.listdir(os.path.join(data_path, class_dir)):
                if os.path.splitext(filename)[1].lower() in ext:
                    try:
                        with Image.open(os.path.join(data_path, class_dir, filename)) as f:
                            f = f.convert('RGB')
                            f.verify()

                    except (ValueError, SyntaxError) as ve:
                        print('Corrupt file:', os.path.join(data_path, class_dir, filename))
                        bad_count +=1
                        if params.delete_bad:
                            os.remove(os.path.join(data_path, class_dir, filename))

                    except (OSError, SyntaxError) as oe:
                        print('Corrupt file:', os.path.join(data_path, class_dir, filename))
                        bad_count +=1
                        if params.delete_bad:
                            os.remove(os.path.join(data_path, class_dir, filename))

            print('Total ' + str(bad_count) + ' corrupt files in ' + class_dir)
            bad_dict[class_dir] = bad_count

    if bad_dict == {}:
        print('\nNo corrupt images detected')
    else:
        print('\nTotal number of corrupt images in each class')
        for key, value in bad_dict.items() :
            print(' ', key, value)



if __name__ == "__main__":
    main()