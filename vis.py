import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils.inceptionv1_caffe import relu_to_redirected_relu
from utils.vis_utils import preprocess, simple_deprocess, load_model, set_seed, mean_loss, ModelPlus, Jitter, register_simple_hook, calc_image_size
from utils.decorrelation import get_decorrelation_layers, RandomScaleLayer, RandomRotationLayer, CenterCropLayer, decorrelate_content
from utils.tile_utils import tile_tensor, rebuild_tensor, get_tiling_info, handle_spectral


def main():
    parser = argparse.ArgumentParser()
    # Input options
    parser.add_argument("-num_classes", type=int, default=120)
    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-layer", type=str, default='mixed5a')
    parser.add_argument("-model_file", type=str, default='')
    parser.add_argument("-channel", type=int, default=-1)
    parser.add_argument("-extract_neuron", action='store_true')
    parser.add_argument("-image_size", type=str, default='224,224')
    parser.add_argument("-content_image", type=str, default='')

    # Output options
    parser.add_argument("-save_iter", type=int, default=0)
    parser.add_argument("-print_iter", type=int, default=25)
    parser.add_argument("-output_image", type=str, default='out.jpg')

    # Optimization options
    parser.add_argument( "-lr", "-learning_rate", type=float, default=1.5)
    parser.add_argument("-num_iterations", type=int, default=500)
    parser.add_argument("-jitter", type=str, default='16')
    parser.add_argument("-fft_decorrelation", action='store_true')
    parser.add_argument("-decay_power", type=float, default=1.0)
    parser.add_argument("-color_decorrelation", help="", nargs="?", type=str, const="none")
    parser.add_argument("-random_scale", help="", nargs="?", type=str, const="none")
    parser.add_argument("-random_rotation", help="", nargs="?", type=str, const="none")
    parser.add_argument("-padding", type=int, default=0)

    # Tiling options
    parser.add_argument("-tile_size", default='0')
    parser.add_argument("-tile_overlap", type=float, default=25.0)
    parser.add_argument("-tile_iter", type=int, default=50)

    # Other options
    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-seed", type=int, default=-1)
    parser.add_argument("-no_branches", action='store_true')
    params = parser.parse_args()
    params.image_size = [int(m) for m in params.image_size.split(',')]
    params.tile_size = [int(m) for m in params.tile_size.split(',')]
    params.tile_size = [params.tile_size[0]] * 2 if len(params.tile_size) == 1 else params.tile_size
    params.tile_overlap = params.tile_overlap / 100 if params.tile_overlap > 1 else params.tile_overlap
    main_func(params)


def main_func(params):
    if params.content_image != '':
        params.image_size = calc_image_size(params.content_image, params.image_size)
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
    mod_list = []
    if params.fft_decorrelation or params.color_decorrelation:
        if params.color_decorrelation == 'none':
            try:
                params.color_decorrelation = torch.load(params.model_file)['color_correlation_svd_sqrt']
            except:
                pass
        d_layers, deprocess_img = get_decorrelation_layers(image_size=params.image_size, input_mean=params.data_mean, device=params.use_device, \
                                                           decorrelate=(params.fft_decorrelation, params.color_decorrelation), decay_power=params.decay_power)
        mod_list += d_layers
    else:
        deprocess_img = None
    if params.padding > 0:
        pad_mod = nn.ReflectionPad2d(params.padding)
        mod_list.append(pad_mod)
    params.jitter = [int(j) for j in params.jitter.split(',')]
    if params.jitter[0] > 0:
        jit_mod = Jitter(params.jitter[0])
        mod_list.append(jit_mod)
    if params.random_scale:
        scale_mod = RandomScaleLayer(params.random_scale)
        mod_list.append(scale_mod)
    if params.random_rotation:
        rot_mod = RandomRotationLayer(params.random_rotation)
        mod_list.append(rot_mod)
    if len(params.jitter) > 1:
        jit_mod_two = Jitter(params.jitter[1])
        mod_list.append(jit_mod_two)
    if params.padding > 0:
        crop_mod = CenterCropLayer(params.padding)
        mod_list.append(crop_mod)
    prep_net = nn.Sequential(*mod_list)

    # Full network
    net = ModelPlus(prep_net, cnn)
    loss_func = mean_loss
    loss_modules = register_simple_hook(net.net, params.layer, params.channel, loss_func=loss_func, neuron=params.extract_neuron)

    # Create input image
    if params.content_image == '':
        if params.fft_decorrelation:
            input_tensor = torch.randn(*((3,) + mod_list[0].freqs_shape)).unsqueeze(0).to(params.use_device) * 0.01
        else:
            input_tensor = torch.randn(3, *params.image_size).unsqueeze(0).to(params.use_device) * 0.01
    else:
        input_tensor = preprocess(params.content_image, params.image_size, params.data_mean, params.not_caffe).to(params.use_device)
        if params.fft_decorrelation != 'none' or params.color_decorrelation != 'none':
            input_tensor = decorrelate_content(input_tensor, mod_list)

    print('Running optimization with ADAM')

    # Create visualization(s)
    if params.tile_size[0] == 0:
        output_tensor = dream(net, input_tensor, params.num_iterations, params.lr, loss_modules, params.save_iter, \
                              params.print_iter, params.output_image, [params.data_mean, params.not_caffe], deprocess_img)
    else:
        filename, ext = os.path.splitext(params.output_image)
        t_size, t_pattern, t_num = get_tiling_info((1,3,*params.image_size), params.tile_size, params.tile_overlap)
        print('\nTile pattern', str(t_pattern).replace(', ', 'x'), '\nNumber of tiles', t_num, '\n')
        is_spectral = True if input_tensor.dim() == 5 else False
        for dream_iter in range(1, params.num_iterations+1):
            input_tensor = handle_spectral(input_tensor, mod_list, params.image_size, params.decay_power) if is_spectral else input_tensor
            tensor_tiles, output_tiles = tile_tensor(input_tensor.clone(), tile_size=params.tile_size, tile_overlap=params.tile_overlap), []
            for i, tile in enumerate(tensor_tiles):
                print('Processing tile', i+1, 'of', len(tensor_tiles))
                tile = handle_spectral(tile, mod_list, params.tile_size, params.decay_power) if is_spectral else tile
                tile = dream(net, tile.clone().detach(), params.tile_iter, params.lr, loss_modules, 0, params.print_iter, 'None', None, None)
                output_tiles.append(tile)

            output_tiles = handle_spectral(output_tiles, mod_list, params.tile_size, params.decay_power) if is_spectral else output_tiles
            output_tensor = rebuild_tensor(output_tiles, input_tensor.size(), tile_size=params.tile_size, tile_overlap=params.tile_overlap)
            output_tensor = handle_spectral(output_tensor, mod_list, params.image_size, params.decay_power) if is_spectral else output_tensor

            if params.save_iter > 0 and dream_iter > 0 and dream_iter % params.save_iter == 0:
                if deprocess_img != None:
                    save_tensor = deprocess_img(output_tensor.clone().detach())
                else:
                    save_tensor = output_tensor.clone().detach()
                simple_deprocess(save_tensor, filename + '_' + str(dream_iter) + ext, params.data_mean, params.not_caffe)

            if params.num_iterations > 1:
                input_tensor = output_tensor.clone()

    if deprocess_img != None:
        output_tensor = deprocess_img(output_tensor)
    simple_deprocess(output_tensor, params.output_image, params.data_mean, params.not_caffe)



# Function to maximize CNN activations
def dream(net, img, iterations, lr, loss_modules, save_iter, print_iter, output_image, deprocess_info, deprocess_img):
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
            if deprocess_img != None:
                simple_deprocess(deprocess_img(img.detach()), filename + '_' + str(i) + \
                                 ext, deprocess_info[0], deprocess_info[1])
            else:
                simple_deprocess(img.detach(), filename + '_' + str(i) + ext, deprocess_info[0], deprocess_info[1])
        optimizer.step()
    return img.detach()



if __name__ == "__main__":
    main()