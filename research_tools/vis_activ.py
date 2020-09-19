import os
import random
import argparse
import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms
from PIL import Image

from utils.inceptionv1_caffe import relu_to_redirected_relu
from utils.vis_utils import simple_deprocess, load_model, register_layer_hook


def main():
    parser = argparse.ArgumentParser()

    # Input options
    parser.add_argument("-num_classes", type=int, default=120)
    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-data_sd", type=str, default='')
    parser.add_argument("-layer", type=str, default='fc')
    parser.add_argument("-model_file", type=str, default='')


    # Other options
    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-no_branches", action='store_true')


    parser.add_argument("-content_image", type=str, default='')
    parser.add_argument("-channels", type=str, default='5')
    parser.add_argument("-channel_mode", choices=['strong', 'avg', 'weak'], default='strong')
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    if 'cuda' in params.use_device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    params.channels = [int(c) for c in params.channels.split(',')]

    cnn, norm_vals, _ = load_model(params.model_file, params.num_classes, has_branches=not params.no_branches)
    if norm_vals != None:
        params.data_mean = norm_vals[0]
        params.data_sd = norm_vals[1]
    else:
        params.data_mean = [float(m) for m in params.data_mean.split(',')]
        params.data_mean.reverse() # RGB to BGR

    relu_to_redirected_relu(cnn)
 
    test_input = preprocess(params.content_image, params.data_mean, params.data_sd).to(params.use_device)

    cnn = cnn.to(params.use_device).eval()
    for param in cnn.parameters():
        params.requires_grad = False

    rank_hooks = []
    for layer_name in params.layer.split(','):
        rank_module = PrintActiv(params.channels, params.channel_mode, params.layer.replace('/', '_'))
        rank_hooks += register_layer_hook(cnn, layer_name, rank_module)

    str_val = 'strongest' if params.channel_mode == 'strong' else ''
    str_val = 'weakest' if params.channel_mode == 'weak' else str_val
    str_val = 'most average' if params.channel_mode == 'avg' else str_val
    print('Printing the ' + str(params.channels) + ' ' + str_val + ' channels for the selected layers')
    cnn(test_input)



def preprocess(image_name, input_mean, input_sd):
    image = Image.open(image_name).convert('RGB')
    Normalize = transforms.Normalize(mean=input_mean, std=input_sd)
    Loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    caffe_transforms = transforms.Compose([transforms.Lambda(lambda x: x*255), transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    tensor = Normalize(caffe_transforms(Loader(image))).unsqueeze(0)
    return tensor


# Define an nn Module to rank channels based on activation strength
class RankChannels(torch.nn.Module):

    def __init__(self, channels=1, channel_mode='strong'):
        super(RankChannels, self).__init__()
        self.channels = channels
        self.channel_mode = channel_mode

    def sort_channels(self, input):
        channel_list = []
        for i in range(input.size(1)):
            channel_list.append(torch.mean(input.clone().squeeze(0).narrow(0,i,1)).item())
        return sorted((c,v) for v,c in enumerate(channel_list))

    def get_middle(self, sequence):
        num = self.channels[0]
        m = (len(sequence) - 1)//2 - num//2
        return sequence[m:m+num]

    def remove_channels(self, cl):
        return [c for c in cl if c[1] not in self.channels]

    def rank_channel_list(self, input):
        top_channels = self.channels[0]
        channel_list = self.sort_channels(input)

        if 'strong' in self.channel_mode:
            channel_list.reverse()
        elif 'avg' in self.channel_mode:
            channel_list = self.get_middle(channel_list)

        channels = []
        for i in range(top_channels):
            channels.append(channel_list[i])
        return channels

    def forward(self, input):
        return self.rank_channel_list(input)


# Define a simple forward hook to collect DeepDream loss for multiple channels
class PrintActiv(torch.nn.Module):
    def __init__(self, channels, mode, name):
        super(PrintActiv, self).__init__()
        self.get_rank = RankChannels(channels, mode)
        self.name = name

    def forward(self, module, input, output):
        channels = self.get_rank(output)
        print('  ', self.name, channels)



if __name__ == "__main__":
    main()