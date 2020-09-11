import os
import shutil
import argparse
import torch
from torchvision import models, transforms
from PIL import Image

from utils.vis_utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_classes", type=int, default=4)

    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-data_sd", type=str, default='')

    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-model_file", type=str, default='')

    parser.add_argument("-unsorted_data", type=str, default='')
    parser.add_argument("-sorted_data", type=str, default='')

    parser.add_argument("-cat", type=int, default=-1)
    parser.add_argument("-confidence_min", type=float, default=-1)
    parser.add_argument("-confidence_max", type=float, default=-1)

    parser.add_argument("-class_strings", type=str, default='')
    params = parser.parse_args()
    main_func(params)


Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


def main_func(params):
    if 'cuda' in params.use_device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    cnn, norm_vals, num_classes = load_model(params.model_file, params.num_classes)
    if norm_vals != None:
        params.data_mean = norm_vals[0]
        params.data_sd = norm_vals[1]
    else:
        params.data_mean = [float(m) for m in params.data_mean.split(',')]
        params.data_sd = [float(m) for m in params.data_sd.split(',')]

    cnn = cnn.to(params.use_device).eval()
    for param in cnn.parameters():
        params.requires_grad = False
    cnn.eval()

    class_strings = params.class_strings.split(',')
    if params.cat != -1:
        try:
            os.mkdir(str(os.path.join(params.sorted_data, str(params.cat))))
        except OSError:
            print ()
    else:
        if params.class_strings == '':
            create_new_dirs(params.sorted_data, num_classes)
        else:
            create_new_class_dirs(params.sorted_data, class_strings)


    class_list = get_sorted_dirs(params.sorted_data)
    print('Sorting images into the following classes:')
    print(' ',class_list)
    ext = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]

    for current_path, dirs, files in os.walk(params.unsorted_data, topdown=True):
        for file in files:
            if os.path.splitext(file)[1].lower() in ext:
                test_input = preprocess(os.path.join(current_path, file), params.data_mean, params.data_sd).to(params.use_device)
                output = cnn(test_input)
                output = output[0] if type(output) == tuple else output
                index = output.argmax().item()

                if params.confidence_min != -1 or params.confidence_max != -1:
                    sm = torch.nn.Softmax(dim=1)
                    probabilities = sm(output)[0][index]
                    if params.confidence_min != -1 and params.confidence_max != -1:
                        confident = True if params.confidence_min < probabilities and probabilities < params.confidence_max else False

                    elif params.confidence_min != -1:
                        confident = True if params.confidence_min < probabilities else False
                    elif params.confidence_max != -1:
                        confident = True if params.confidence_max > probabilities else False
                else:
                    confident = True

                if index == params.cat and confident or params.cat == -1 and confident:
                    if params.cat != -1 and index == params.cat:
                        new_path = os.path.join(params.sorted_data, str(params.cat))
                    else:
                        if params.class_strings == '':
                             new_path = os.path.join(params.sorted_data, str(index))
                        else:
                             new_path = os.path.join(params.sorted_data, class_strings[index])
                    print(index, file)

                    try:
                        shutil.copy2(os.path.join(os.path.normpath(current_path), file), os.path.join(new_path, file))
                    except (OSError, SyntaxError) as oe:
                        print('Failed:', os.path.join(os.path.normpath(current_path), file))



def create_new_dirs(sorted_path, num_classes):
    for c_dir in range(num_classes):
        try:
            os.mkdir(os.path.join(sorted_path, str(c_dir)))
        except OSError:
            pass


def create_new_class_dirs(sorted_path, class_strings):
    for c_dir in class_strings:
        try:
            os.mkdir(os.path.join(sorted_path, c_dir))
        except OSError:
            pass


def get_sorted_dirs(data_path):
    classes = [d.name for d in os.scandir(data_path) if d.is_dir()]
    classes.sort()
    return classes


def preprocess(image_name, input_mean, input_sd):
    image = Image.open(image_name).convert('RGB')
    Normalize = transforms.Normalize(mean=input_mean, std=input_sd)
    Loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    caffe_transforms = transforms.Compose([transforms.Lambda(lambda x: x*255), transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    tensor = Normalize(caffe_transforms(Loader(image))).unsqueeze(0)
    return tensor



if __name__ == "__main__":
    main()
