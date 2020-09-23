import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from utils.inceptionv1_caffe import relu_to_redirected_relu
from utils.vis_utils import simple_deprocess, load_model, set_seed, mean_loss, ModelPlus, Jitter, register_layer_hook
from utils.decorrelation import get_decorrelation_layers, RandomScaleLayer


def main():
    parser = argparse.ArgumentParser()

    # Input options
    parser.add_argument("-num_classes", type=int, default=120)
    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-layer", type=str, default='fc')
    parser.add_argument("-model_file", type=str, default='')
    parser.add_argument("-image_size", type=str, default='224,224')

    # Output options
    parser.add_argument("-model_epoch", type=int, default=10)
    parser.add_argument("-save_iter", type=int, default=0)
    parser.add_argument("-print_iter", type=int, default=25)
    parser.add_argument("-output_dir", type=str, default='')

    # Optimization options
    parser.add_argument( "-lr", "-learning_rate", type=float, default=1.5)
    parser.add_argument("-num_iterations", type=int, default=500)
    parser.add_argument("-jitter", type=int, default=32)
    parser.add_argument("-fft_decorrelation", action='store_true')
    parser.add_argument("-color_decorrelation", help="", nargs="?", type=str, const="none")
    parser.add_argument("-random_scale", nargs="?", type=str, const="none")

    # Other options
    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-seed", type=int, default=-1)
    parser.add_argument("-no_branches", action='store_true')

    # Batch
    parser.add_argument("-batch_size", type=int, default=10)
    parser.add_argument("-start_channel", type=int, default=-1)
    parser.add_argument("-end_channel", type=int, default=-1)
    params = parser.parse_args()

    params.image_size = [int(m) for m in params.image_size.split(',')]
    main_func(params)


def main_func(params):
    if params.seed > -1:
        set_seed(params.seed)

    if 'cuda' in params.use_device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    try:
        model_epoch = torch.load(params.model_file, map_location='cpu')['epoch']
    except:
        model_epoch = params.model_epoch

    cnn, norm_vals, _ = load_model(params.model_file, params.num_classes, has_branches=not params.no_branches)
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
    if params.fft_decorrelation:
        if params.color_decorrelation == 'none':
            try:
                params.color_decorrelation = torch.load(params.model_file)['color_correlation_svd_sqrt']
            except:
                pass
        d_layers, deprocess_img = get_decorrelation_layers(image_size=params.image_size, input_mean=params.data_mean, device=params.use_device, \
                                                           decorrelate=(params.fft_decorrelation, params.color_decorrelation))
        mod_list += d_layers
    else:
        deprocess_img = None
    if params.random_scale:
        scale_mod = RandomScaleLayer(params.random_scale)
        mod_list.append(scale_mod)
    if params.jitter > 0:
        jit_mod = Jitter(params.jitter)
        mod_list.append(jit_mod)
    prep_net = nn.Sequential(*mod_list)

    # Full network
    net = ModelPlus(prep_net, cnn)

    # Create basic input
    if params.fft_decorrelation:
        input_tensor = torch.randn(*((3,) + mod_list[0].freqs_shape)).to(params.use_device) * 0.01
    else:
        input_tensor = torch.randn(3, *params.image_size).to(params.use_device) * 0.01

    # Determine how many visualizations to generate
    num_channels, layer_dim = get_num_channels(deepcopy(cnn), params.layer, input_tensor.detach())

    # Loss module setup
    loss_func = mean_loss
    loss_modules = register_hook_batch_selective(net.net, params.layer, loss_func=loss_func)
    loss_modules[0].layer_dim = layer_dim

    # Stack basic inputs into batch
    input_tensor_list = []
    for t in range(params.batch_size):
        input_tensor_list.append(input_tensor.clone())
    input_tensor = torch.stack(input_tensor_list)

    output_basename = os.path.join(params.output_dir, params.layer.replace('/', '_'))

    num_channels = num_channels if params.end_channel < 0 else params.end_channel
    start_val = 0 if params.start_channel < 0 else params.start_channel
    vis_count = start_val

    num_channels_vis = len(range(start_val, num_channels))
    num_runs = -(-num_channels_vis // params.batch_size)

    print('\nVisualizing ' + str(num_channels) + ' ' + params.layer + ' channels')
    print('Running optimization with ADAM\n')

    for num_vis in range(num_runs):
        print('Processing batch number ' + str(num_vis + 1) + '/' + str(num_runs))
        loss_modules[0].channel_end += params.batch_size
        if loss_modules[0].channel_end > num_channels - 1:
            loss_modules[0].channel_end = num_channels

        batch_count = len(range(loss_modules[0].channel_start, loss_modules[0].channel_end))
        if batch_count < params.batch_size:
            if params.fft_decorrelation:
                input_tensor = input_tensor[:batch_count,:,:,:,:]
            else:
                input_tensor = input_tensor[:batch_count,:,:,:]

        output_tensor = dream(net, input_tensor.clone(), params.num_iterations, params.lr, loss_modules, params.print_iter)

        if deprocess_img != None:
            output_tensor = deprocess_img(output_tensor)

        for batch_val in range(params.batch_size):
            simple_deprocess(output_tensor[batch_val], output_basename + '_c' + str(vis_count).zfill(4) + '_e' + str(model_epoch).zfill(3) + \
                             '.jpg', params.data_mean, params.not_caffe)
            vis_count += 1
            if vis_count > num_channels or batch_val == batch_count - 1:
                break

        loss_modules[0].channel_start += params.batch_size



# Function to maximize CNN activations
def dream(net, img, iterations, lr, loss_modules, print_iter):
    img = nn.Parameter(img)
    optimizer = torch.optim.Adam([img], lr=lr)

    # Training loop
    for i in range(1, iterations + 1):
        optimizer.zero_grad()
        net(img)
        loss = loss_modules[0].loss
        loss.backward()

        if print_iter > 0 and i % print_iter == 0:
            print('  Iteration', str(i) + ',', 'Loss', str(loss.item()))

        optimizer.step()
    return img.detach()



class ChannelRecorder(torch.nn.Module):
    def forward(self, module, input, output):
        self.size = list(output.size())


# Determine total number of channels
def get_num_channels(test_net, layer, test_tensor):
    if test_tensor.dim() > 3:
        test_tensor = test_tensor[:,:,:,1]
    get_channels = ChannelRecorder()
    channel_catcher = register_layer_hook(test_net, layer, get_channels)
    with torch.no_grad():
        test_net(test_tensor.unsqueeze(0))
    num_channels = channel_catcher[0].size
    return num_channels[1], len(num_channels)


def register_hook_batch_selective(net, layer_name, loss_func=mean_loss):
    loss_module = SimpleDreamLossHookChannels(loss_func)
    return register_layer_hook(net, layer_name, loss_module)


# Define a simple forward hook to collect DeepDream loss for multiple channels
class SimpleDreamLossHookChannels(torch.nn.Module):
    def __init__(self, loss_func=mean_loss):
        super(SimpleDreamLossHookChannels, self).__init__()
        self.get_loss = loss_func
        self.channel_start = 0
        self.channel_end = 0
        self.layer_dim = 2
        self.get_neuron = False

    def forward(self, module, input, output):
        output = self.extract_neuron(output) if self.get_neuron == True else output
        vis_list = list(range(self.channel_start, self.channel_end))
        loss = 0
        for i in range(0, len(vis_list)):
            if self.layer_dim == 2:
                loss = loss + self.get_loss(output[i, vis_list[i]])
            elif self.layer_dim == 4:
                loss = loss + self.get_loss(output[i, vis_list[i], :, :])
        self.loss = -loss

    def extract_neuron(self, input):
        x = input.size(2) // 2
        y = input.size(3) // 2
        return input[:, :, y:y+1, x:x+1]



if __name__ == "__main__":
    main()