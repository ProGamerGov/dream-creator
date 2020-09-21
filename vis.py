import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils.training_utils import save_csv_data
from utils.inceptionv1_caffe import relu_to_redirected_relu
from utils.vis_utils import preprocess, simple_deprocess, load_model, set_seed, mean_loss, ModelPlus, Jitter, register_simple_hook
from utils.decorrelation import get_decorrelation_layers


def main():
    parser = argparse.ArgumentParser()
    # Input options
    parser.add_argument("-num_classes", type=int, default=120)
    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-layer", type=str, default='mixed5a')
    parser.add_argument("-model_file", type=str, default='')
    parser.add_argument("-channel", type=int, default=-1)
    parser.add_argument("-center_neuron", action='store_true')
    parser.add_argument("-image_size", type=str, default='224,224')
    parser.add_argument("-content_image", type=str, default='')

    # Output options
    parser.add_argument("-save_iter", type=int, default=0)
    parser.add_argument("-print_iter", type=int, default=25)
    parser.add_argument("-output_image", type=str, default='out.jpg')

    # Optimization options
    parser.add_argument( "-lr", "-learning_rate", type=float, default=1.5)
    parser.add_argument("-num_iterations", type=int, default=500)
    parser.add_argument("-jitter", type=int, default=32)

    # Other options
    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-seed", type=int, default=-1)
    parser.add_argument("-no_branches", action='store_true')
    
    parser.add_argument("-fft_decorrelation", action='store_true')
    params = parser.parse_args()
    params.image_size = [int(m) for m in params.image_size.split(',')]
    main_func(params)


def main_func(params):
    if params.seed > -1:
        set_seed(params.seed)

    if 'cuda' in params.use_device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    cnn, norm_vals, num_classes = load_model(params.model_file, params.num_classes, has_branches=not params.no_branches)
    if norm_vals != None and params.data_mean == '':
        params.data_mean = norm_vals[0]
    else:
        params.data_mean = [float(m) for m in params.data_mean.split(',')]

    relu_to_redirected_relu(cnn)

    cnn = cnn.to(params.use_device).eval()
    for param in cnn.parameters():
        params.requires_grad = False

    # Preprocessing net layers
    jit_mod = Jitter(params.jitter)
    mod_list = []
    if params.fft_decorrelation:    
        mod_list += get_decorrelation_layers(params.image_size, input_mean=params.data_mean, device=params.use_device)
    mod_list.append(jit_mod)
    prep_net = nn.Sequential(*mod_list)

    # Full network
    net = ModelPlus(prep_net, cnn)
    loss_func = mean_loss
    loss_modules = register_simple_hook(net.net, params.layer, params.channel, loss_func=loss_func, neuron=params.center_neuron)

    if params.content_image == '':
        if params.fft_decorrelation:
            init_val_size = (1, 3) + mod_list[0].freqs_shape + (2,)
            input_tensor = (torch.randn(*init_val_size) * 0.01).to(params.use_device)
        else:
            input_tensor = torch.randn(3,*params.image_size).unsqueeze(0).to(params.use_device) * 0.01
    else:
        input_tensor = preprocess(params.content_image, params.image_size, params.data_mean, params.not_caffe).to(params.use_device)

    print('Running optimization with ADAM')

    # Create 224x224 image
    output_tensor = dream(net, input_tensor, params.num_iterations, params.lr, loss_modules, params.data_mean, \
                          params.save_iter, params.print_iter, params.output_image, params.not_caffe)
    if params.fft_decorrelation:
        output_tensor = mod_list[1](mod_list[0](output_tensor))
    simple_deprocess(output_tensor, params.output_image, params.data_mean, params.not_caffe)


# Function to maximize CNN activations
def dream(net, img, iterations, lr, loss_modules, m, save_iter, print_iter, output_image, not_caffe):
    filename, ext = os.path.splitext(output_image)
    img = nn.Parameter(img)
    optimizer = torch.optim.Adam([img], lr=lr)

    # Training loop
    for i in range(1, iterations + 1):
        optimizer.zero_grad()
        net(img)
        loss = loss_modules[0].loss
        loss.backward()

        if print_iter > 0 and i % print_iter == 0:
            print('Iteration', str(i) + ',', 'Loss', str(loss.item()))

        if save_iter > 0 and i > 0 and i % save_iter == 0:
            simple_deprocess(img.detach(), filename + '_' + str(i) + ext, m, not_caffe)
        optimizer.step()
    return img.detach()



if __name__ == "__main__":
    main()
