import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.inceptionv1_caffe import InceptionV1_Caffe


# Simple preprocess
def preprocess(image_name, image_size, input_mean, not_caffe):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple and type(image_size) is not list:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    if not not_caffe:
        tensor_transforms = transforms.Compose([transforms.Lambda(lambda x: x*255),
                                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
                                                transforms.Normalize(mean=input_mean, std=[1,1,1])
                                               ])
        tensor = tensor_transforms(Loader(image)).unsqueeze(0)
    else:
        Normalize = transforms.Normalize(mean=input_mean, std=[1,1,1])
        tensor = Normalize(Loader(image)).unsqueeze(0)
    return tensor


# Simple Deprocess
def simple_deprocess(output_tensor, output_name, input_mean, not_caffe):
    save_name = output_name
    input_mean = [n * -1 for n in input_mean]

    if not not_caffe:
        tensor_transforms = transforms.Compose([transforms.Normalize(mean=input_mean, std=[1,1,1]),
                                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
                                                transforms.Lambda(lambda x: x/255),
                                               ])
        output_tensor = tensor_transforms(output_tensor.squeeze(0).cpu())
    else:
        Normalize = transforms.Normalize(mean=input_mean, std=[1,1,1])
        output_tensor = Normalize(output_tensor.squeeze(0).cpu())
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor)
    image.save(save_name)


# Build the model definition
def build_model(model_file='pt_bvlc.pth', mode='bvlc', num_classes=120, load_branches=True):
    base_list = {'pt_bvlc.pth': (1000, 'bvlc'), 'pt_places365.pth': (365, 'p365'), 'pt_inception5h.pth': (1008, '5h')}
    base_name = os.path.basename(model_file)
    if base_name.lower() in base_list:
        load_classes, mode = base_list[base_name.lower()]
        if mode == '5h':
            load_branches = False
    else:
        load_classes = num_classes

    cnn = InceptionV1_Caffe(load_classes, mode=mode, load_branches=load_branches)

    if base_name.lower() not in base_list:
        cnn.replace_fc(load_classes, load_branches)
    return cnn


def load_model(model_file, num_classes=120, has_branches=True, mode='bvlc'):
    checkpoint = torch.load(model_file, map_location='cpu')

    # Attempt to load model class info
    try:
        base_model = checkpoint['base_model']
    except:
        base_model = None
    mode = mode if base_model == None else base_model

    try:
        loaded_classes = checkpoint['num_classes']
    except:
        loaded_classes = None
    num_classes = num_classes if loaded_classes == None else loaded_classes

    try:
        norm_vals = checkpoint['normalize_params']
    except:
        norm_vals = None
    try:
        h_branches = checkpoint['has_branches']
    except:
        h_branches = None
    has_branches = has_branches if h_branches == None else h_branches

    cnn = build_model(model_file, mode, num_classes, load_branches=has_branches)

    if type(checkpoint) == dict:
        model_keys = checkpoint.keys()
        cnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_name = os.path.basename(model_file)
        if base_name.lower() == 'pt_bvlc.pth' or base_name.lower() == 'pt_places365.pth':
            cnn.use_fc = False
        if base_name.lower() == 'pt_bvlc.pth' or base_name.lower() == 'pt_inception5h.pth':
            norm_vals = [[103.939,116.779,123.68], [1,1,1], 'BGR']
        elif base_name.lower() == 'pt_places365.pth':
            norm_vals = [[104.051,112.514,116.676], [1,1,1], 'BGR']
        cnn.load_state_dict(checkpoint)
    return cnn, norm_vals, num_classes


# Set global seeds to make output reproducible
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)


# Basic mean loss
def mean_loss(input):
    return input.mean()


# Run preprocessing net before model
class ModelPlus(torch.nn.Module):

    def __init__(self, input_net, net):
        super(ModelPlus, self).__init__()
        self.input_net = input_net
        self.net = net

    def forward(self, input):
        return self.net(self.input_net(input))


class Jitter(torch.nn.Module):

    def __init__(self, jitter_val):
        super(Jitter, self).__init__()
        self.jitter_val = jitter_val

    def roll_tensor(self, input):
        h_shift = random.randint(-self.jitter_val, self.jitter_val)
        w_shift = random.randint(-self.jitter_val, self.jitter_val)
        return torch.roll(torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3)

    def forward(self, input):
        return self.roll_tensor(input)


# Create loss module and hook for multiple channels
def register_hook_batch(net, layer_name, loss_func=mean_loss):
    loss_module = SimpleDreamLossHookBatch(loss_func)
    return register_layer_hook(net, layer_name, loss_module)

# Create loss module and hook
def register_simple_hook(net, layer_name, channel=-1, loss_func=mean_loss, mode='loss', neuron=False):
    loss_module = SimpleDreamLossHook(channel, loss_func, mode, neuron)
    return register_layer_hook(net, layer_name, loss_module)


# Create layer hook
def register_layer_hook(net, layer_name, loss_module):
    layer_name = layer_name.replace('\\', '/').replace('.', '/')
    if len(layer_name.split('/')) == 1:
        getattr(net, layer_name).register_forward_hook(loss_module)
    elif len(layer_name.split('/')) == 2:
        layer_name = layer_name.split('/')
        getattr(getattr(net, layer_name[0]), layer_name[1]).register_forward_hook(loss_module)
    return [loss_module]


# Define a simple forward hook to collect DeepDream loss for multiple channels
class SimpleDreamLossHookBatch(torch.nn.Module):
    def __init__(self, loss_func=mean_loss):
        super(SimpleDreamLossHookBatch, self).__init__()
        self.get_loss = loss_func

    def forward(self, module, input, output):
        loss = 0
        for batch in range(output.size(0)):
            loss = loss + self.get_loss(output[batch, batch])
        self.loss = -loss


# Define a simple forward hook to collect DeepDream loss
class SimpleDreamLossHook(torch.nn.Module):
    def __init__(self, channel=-1, loss_func=mean_loss, mode='loss', neuron=False):
        super(SimpleDreamLossHook, self).__init__()
        self.channel = channel
        self.get_loss = loss_func
        self.mode = mode
        self.neuron = neuron

    def get_neuron(self, input):
        x = input.size(2) // 2
        y = input.size(3) // 2
        return input[:, :, y:y+1, x:x+1]

    def forward_loss(self, input):
        self.loss = -self.get_loss(input)

    def forward_feature(self, input):
        self.feature = input

    def forward(self, module, input, output):
        if self.channel > -1:
            if self.neuron:
                output = self.get_neuron(output)
            output = output[:,self.channel]
        if self.mode == 'loss':
            self.forward_loss(output)
        elif self.mode == 'feature':
            self.forward_feature(output)