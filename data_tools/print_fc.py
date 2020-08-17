import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-print_string", action='store_true')
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    classes = [d.name for d in os.scandir(params.data_path) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print('Classes:')
    print('', class_to_idx)

    if params.print_string:
        print('\nParam string:')
        class_string = ','.join(classes)
        print('', class_string)



if __name__ == "__main__":
    main()