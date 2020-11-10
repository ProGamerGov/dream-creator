import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper function for returning deprocessing of decorrelated tensors
def get_decorrelation_layers(image_size=(224,224), input_mean=[1,1,1], device='cpu', decay_power=0.75, decorrelate=[True,None]):
    mod_list = []
    if decorrelate[0] == True:
        spatial_mod = SpatialDecorrelationLayer(image_size, decay_power=decay_power, device=device)
        mod_list.append(spatial_mod)
    if decorrelate[1] != None:
        if torch.is_tensor(decorrelate[1]):
            matrix = decorrelate[1]
        else:
            matrix = 'imagenet' if decorrelate[1] == 'none' or decorrelate[1].lower() == 'imagenet' else decorrelate[1]
        color_mod = ColorDecorrelationLayer(correlation_matrix=matrix, device=device)
        mod_list.append(color_mod)
    transform_mod = TransformLayer(input_mean=input_mean, device=device)
    mod_list.append(transform_mod)

    if decorrelate[0] == True and decorrelate[1] == None:
        deprocess_img = lambda x: transform_mod.forward(spatial_mod.forward(x))
    elif decorrelate[0] == False and decorrelate[1] != None:
        deprocess_img = lambda x: transform_mod.forward(color_mod.forward(x))
    elif decorrelate[0] == True and decorrelate[1] != None:
        deprocess_img = lambda x: transform_mod.forward(color_mod.forward(spatial_mod.forward(x)))
    return mod_list, deprocess_img


# Helper function to decorrelate content image
def decorrelate_content(content_image, mod_list):
    s, c, t = None, None, None
    for i, mod in enumerate(mod_list):
        if isinstance(mod, SpatialDecorrelationLayer):
            s = i
        if isinstance(mod, ColorDecorrelationLayer):
            c = i
        if isinstance(mod, TransformLayer):
            t = i

    if t != None:
        content_image = mod_list[t].untransform(content_image)
        mod_list[t].activ = lambda x: x.clamp(0,1)
    if c != None:
        content_image = mod_list[c].decorrelate_color(content_image)
    if s != None:
        content_image = mod_list[s].fft_image(content_image)
    return content_image


# Spatial Decorrelation layer based on tensorflow/lucid & greentfrapp/lucent
class SpatialDecorrelationLayer(torch.nn.Module):

    def __init__(self, image_size=(224,224), decay_power=1.0, device='cpu'):
        super(SpatialDecorrelationLayer, self).__init__()
        self.setup_scale(image_size, decay_power, device)

    def setup_scale(self, image_size, decay_power=1.0, device='cpu'):
        self.h, self.w = image_size[0], image_size[1]
        self.scale = self.create_scale(image_size, decay_power).to(device)

    def create_scale(self, size, decay_power=1.0):
        freqs = SpatialDecorrelationLayer.rfft2d_freqs(*size)
        self.freqs_shape = freqs.size() + (2,)
        scale = 1.0 / torch.max(freqs, torch.full_like(freqs, 1.0 / max(size))) ** decay_power
        scale = scale * ((size[0] * size[1]) ** (1 / 2))
        return scale[None, None, ..., None]

    @staticmethod
    def rfft2d_freqs(h, w):
        fy = SpatialDecorrelationLayer.pytorch_fftfreq(h)[:, None]
        wadd = 2 if w % 2 == 1 else 1
        fx = SpatialDecorrelationLayer.pytorch_fftfreq(w)[: w // 2 + wadd]
        return torch.sqrt((fx * fx) + (fy * fy))

    @staticmethod
    def pytorch_fftfreq(v, d=1.0):
        results = torch.empty(v)
        s = (v - 1) // 2 + 1
        results[:s] = torch.arange(0, s)
        results[s:] = torch.arange(-(v // 2), 0)
        return results * (1.0 / (v * d))

    def fft_image(self, input):
        input = torch.rfft(input, 2, normalized=False)
        return input / self.scale

    def ifft_image(self, input):
        input = input * self.scale
        return torch.irfft(input, 2, normalized=False, signal_sizes=(self.h, self.w))

    def forward(self, input):
        return self.ifft_image(input)


# Color Decorrelation layer based on tensorflow/lucid & greentfrapp/lucent
class ColorDecorrelationLayer(nn.Module):

    def __init__(self, correlation_matrix='imagenet', device='cpu'):
        super(ColorDecorrelationLayer, self).__init__()
        self.color_correlation_n = self.color_correlation_normalized(correlation_matrix).to(device)

    def get_matrix(self, matrix='imagenet'):
        if torch.is_tensor(matrix):
            color_correlation_svd_sqrt = matrix
        elif ',' in matrix:
            m = [float(mx) for mx in matrix.replace('n','-').split(',')]
            color_correlation_svd_sqrt = torch.Tensor([[m[0], m[1], m[2]],
                                                      [m[3], m[4], m[5]],
                                                      [m[6], m[7], m[8]]])
        elif matrix.lower() == 'imagenet':
            color_correlation_svd_sqrt = torch.Tensor([[0.26, 0.09, 0.02],
                                                      [0.27, 0.00, -0.05],
                                                      [0.27, -0.09, 0.03]])
        elif matrix.lower() == 'places365':
            raise NotImplementedError
        return color_correlation_svd_sqrt

    def color_correlation_normalized(self, matrix):
        color_correlation_svd_sqrt = self.get_matrix(matrix)
        max_norm_svd_sqrt = torch.max(color_correlation_svd_sqrt.norm(0))
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
        return color_correlation_normalized.T

    def decorrelate_color(self, input):
        inverse = torch.inverse(self.color_correlation_n)
        return torch.matmul(input.permute(0,2,3,1), inverse).permute(0,3,1,2)

    def forward(self, input):
        return torch.matmul(input.permute(0,2,3,1), self.color_correlation_n).permute(0,3,1,2)


# Preprocess input after decorrelation
class TransformLayer(torch.nn.Module):

    def __init__(self, input_mean=[1,1,1], input_sd=[1,1,1], r=255, device='cpu'):
        super(TransformLayer, self).__init__()
        self.input_mean = torch.as_tensor(input_mean).view(3, 1, 1).to(device)
        self.input_sd = torch.as_tensor(input_sd).view(3, 1, 1).to(device)
        self.r = r
        self.activ = lambda x: torch.sigmoid(x)

    def untransform(self, input):
        input = (input + self.input_mean) * self.input_sd
        return input / self.r

    def forward(self, input):
        input = self.activ(input) * self.r
        return (input - self.input_mean) / self.input_sd


# Randomly scale an input
class RandomScaleLayer(torch.nn.Module):

    def __init__(self, scale_list=(1, 0.975, 1.025, 0.95, 1.05)):
        super(RandomScaleLayer, self).__init__()
        scale_list = (1, 0.975, 1.025, 0.95, 1.05) if scale_list == 'none' else scale_list
        scale_list = [float(s) for s in scale_list.split(',')] if ',' in scale_list else scale_list
        self.scale_list = scale_list

    def get_scale_mat(self, m, device, dtype):
        return torch.tensor([[m, 0.0, 0.0], [0.0, m, 0.0]], device=device, dtype=dtype)

    def rescale_tensor(self, x, scale):
        scale_matrix = self.get_scale_mat(scale, x.device, x.dtype)[None, ...].repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(scale_matrix, x.size())
        return F.grid_sample(x, grid)

    def forward(self, input):
        n = random.randint(0, len(self.scale_list)-1)
        return self.rescale_tensor(input, scale=self.scale_list[n])


# Randomly rotate a tensor from a list of degrees
class RandomRotationLayer(torch.nn.Module):

    def __init__(self, range_degrees=5):
        super(RandomRotationLayer, self).__init__()
        range_degrees = '5' if range_degrees == 'none' else range_degrees
        if range_degrees is not int and ',' in range_degrees:
            self.angle_range = [int(r) for r in range_degrees.replace('n','-').split(',')]
        else:
            self.angle_range = list(range(-int(range_degrees), int(range_degrees) + 1))

    def get_random_angle(self):
        n = random.randint(0, len(self.angle_range) -1)
        return self.angle_range[n] * 3.141592653589793 / 180

    def get_rot_mat(self, theta, device, dtype):
        theta = torch.tensor(theta, device=device, dtype=dtype)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]], device=device, dtype=dtype)

    def rotate_tensor(self, x, theta):
        rotation_matrix = self.get_rot_mat(theta, x.device, x.dtype)[None, ...].repeat(x.shape[0],1,1)
        grid = F.affine_grid(rotation_matrix, x.size())
        return F.grid_sample(x, grid)

    def forward(self, input):
        rnd_angle = self.get_random_angle()
        return self.rotate_tensor(input, rnd_angle)


# Crop the padding off a tensor
class CenterCropLayer(torch.nn.Module):

    def __init__(self, crop_val=0):
        super(CenterCropLayer, self).__init__()
        self.crop_val = crop_val

    def forward(self, input):
        h, w = input.size(2), input.size(3)
        h_crop = input.size(2) - self.crop_val
        w_crop = input.size(3) - self.crop_val
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        return input[:, :, sh:sh + h_crop, sw:sw + w_crop]
