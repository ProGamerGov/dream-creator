import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

    # Other options
    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-seed", type=int, default=-1)
    parser.add_argument("-no_branches", action='store_true')

    parser.add_argument("-fft_decorrelation", action='store_true')
    parser.add_argument("-color_decorrelation", help="", nargs="?", type=str, const="none")
    parser.add_argument("-random_scale", nargs="?", type=str, const="none")

    # Batch
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-channel", type=int, default=0)
    parser.add_argument("-extract_neuron", action='store_true')
    parser.add_argument("-similarity_penalty", type=float, default=1e2)
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

    # Loss module setup
    loss_func = mean_loss
    loss_modules = register_hook_batch_diverse(net=net.net, layer_name=params.layer, loss_func=loss_func, channel=params.channel, penalty_strength=params.similarity_penalty, neuron=params.extract_neuron)

    # Stack basic inputs into batch
    input_tensor_list = []
    for t in range(params.batch_size):
        input_tensor_list.append(input_tensor.clone())
    input_tensor = torch.stack(input_tensor_list)

    output_basename = os.path.join(params.output_dir, params.layer.replace('/', '_'))

    print('\nAttempting to extract ' + str(params.batch_size) + ' different features from ' + params.layer + ' channel ' + str(params.channel))
    print('Running optimization with ADAM\n')
    output_tensor = dream(net, input_tensor.clone(), params.num_iterations, params.lr, loss_modules, params.print_iter)

    if deprocess_img != None:
        output_tensor = deprocess_img(output_tensor)

    for batch_val in range(params.batch_size):
        simple_deprocess(output_tensor[batch_val], output_basename + '_c' + str(params.channel).zfill(4) + '_f' + str(batch_val).zfill(3)  + '_e' + str(model_epoch).zfill(3) + \
                         '.jpg', params.data_mean, params.not_caffe)



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


def register_hook_batch_diverse(net, layer_name, loss_func=mean_loss, channel=0, penalty_strength=1e2, neuron=False):
    loss_module = SimpleDreamLossHookDiverse(loss_func, channel, penalty_strength, neuron)
    return register_layer_hook(net, layer_name, loss_module)


# Define a simple forward hook to collect DeepDream loss for multiple channels
class SimpleDreamLossHookDiverse(torch.nn.Module):
    def __init__(self, loss_func=mean_loss, channel=0, penalty_strength=1e2, neuron=False):
        super(SimpleDreamLossHookDiverse, self).__init__()
        self.get_loss = loss_func
        self.get_neuron = neuron
        self.channel = channel
        self.penalty_strength = penalty_strength

    def forward(self, module, input, output):
        output = self.extract_neuron(output) if self.get_neuron == True else output
        if self.channel != -1:
            loss = -self.get_loss(output[:,self.channel])
        else:
            loss = -self.get_loss(output)
        self.loss = loss - (self.penalty_strength * diversity(output.clone()))

    def extract_neuron(self, input):
        x = input.size(2) // 2
        y = input.size(3) // 2
        return input[:, :, y:y+1, x:x+1]


# Separate channel into it's parts, based on tensorflow/lucid & greentfrapp/lucent
def diversity(input):
    return -sum([ sum([(torch.cosine_similarity(input[j].view(1,-1), input[i].view(1,-1))).sum() for i in range(input.size(0)) if i != j]) \
           for j in range(input.size(0))]) / input.size(0)



if __name__ == "__main__":
    main()
