import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils.training_utils import save_csv_data
from utils.inceptionv1_caffe import relu_to_redirected_relu
from utils.vis_utils import simple_deprocess, load_model, set_seed, mean_loss, ModelPlus, Jitter, register_hook_batch


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
    parser.add_argument("-save_csv", action='store_true')
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

    cnn, norm_vals, num_classes = load_model(params.model_file, params.num_classes, has_branches=params.no_branches)
    if norm_vals != None:
        params.data_mean = norm_vals[0]
    else:
        params.data_mean = [float(m) for m in params.data_mean.split(',')]
        params.data_mean.reverse() # RGB to BGR

    relu_to_redirected_relu(cnn)

    cnn = cnn.to(params.use_device).eval()
    for param in cnn.parameters():
        params.requires_grad = False

    # Preprocessing net layers
    jit_mod = Jitter(params.jitter)
    mod_list = []
    mod_list.append(jit_mod)
    prep_net = nn.Sequential(*mod_list)

    # Full network
    net = ModelPlus(prep_net, cnn)
    loss_func = mean_loss
    loss_modules = register_hook_batch(net.net, params.layer, loss_func=loss_func)

    input_tensor = torch.randn(3,224,224).to('cuda:0') * 0.01
    input_tensor_list = []
    for t in range(num_classes):
        input_tensor_list.append(input_tensor.clone())
    input_tensor = torch.stack(input_tensor_list)

    print('Visualizing ' + str(num_classes) + ' ' + params.layer + ' channels')
    print('Running optimization with ADAM')

    output_basename = os.path.join(params.output_dir, params.layer)

    # Create 224x224 image
    output_tensor = dream(net, input_tensor, params.num_iterations, params.lr, loss_modules, params.data_mean, \
                          params.save_iter, params.print_iter, params.not_caffe, model_epoch, output_basename, params.save_csv)
    for batch in range(output_tensor.size(0)):
        simple_deprocess(output_tensor[batch], output_basename + '_c' + str(batch).zfill(2) + '_e' + str(model_epoch).zfill(3) + \
                         '.jpg', params.data_mean, params.not_caffe)


# Function to maximize CNN activations
def dream(net, img, iterations, lr, loss_modules, m, save_iter, print_iter, use_caffe, epoch, output_basename, should_save_csv):
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
        if should_save_csv:
            save_csv_data('e'+ str(epoch).zfill(3) + '_visloss.txt', [str(i+1), str(loss.item())])

        if save_iter > 0 and i > 0 and i % save_iter == 0:
            for batch in range(img.size(0)):
                simple_deprocess(img[batch].detach(), output_basename + '_c' + str(batch).zfill(2) + '_e' + str(epoch).zfill(3) + '_' + str(i).zfill(4) + '.jpg', m, use_caffe)
        optimizer.step()
    return img.detach()



if __name__ == "__main__":
    main()