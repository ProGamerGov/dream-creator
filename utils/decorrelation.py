import torch
import torch.nn as nn


# Helper function for returning deprocessing of decorrelated tensors
def get_decorrelation_layers(image_size=(224,224), input_mean=[1,1,1], device='cpu', decay_power=0.75, decorrelate=[True,False]):
    mod_list = []
    if decorrelate[0] == True:
        spatial_mod = SpatialDecorrelationLayer(image_size, decay_power=decay_power, device=device)
        mod_list.append(spatial_mod)
    if decorrelate[1] != False:
        matrix = 'imagenet' if decorrelate[1] == '' or decorrelate[1].lower() == 'imagenet' else decorrelate[1]
        color_mod = ColorDecorrelationLayer(correlation_matrix=matrix, device=device)
        mod_list.append(color_mod)
    if decorrelate[0] == True or decorrelate[1] == True:
        transform_mod = TransformLayer(input_mean=input_mean, device=device)
        mod_list.append(transform_mod)
 
    if decorrelate[0] == True and decorrelate[1] == False:
        deprocess_img = lambda x: transform_mod.forward(spatial_mod.forward(x)) 
    elif decorrelate[0] == False and decorrelate[1] != False:
        deprocess_img = lambda x: transform_mod.forward(color_mod.forward(x)) 
    elif decorrelate[0] == True and decorrelate[1] != False:
        deprocess_img = lambda x: transform_mod.forward(color_mod.forward(spatial_mod.forward(x))) 
    return mod_list, deprocess_img


# Spatial Decorrelation layer based on tensorflow/lucid & greentfrapp/lucent
class SpatialDecorrelationLayer(torch.nn.Module):

    def __init__(self, image_size=(224,224), decay_power=1.0, device='cpu'):
        super(SpatialDecorrelationLayer, self).__init__()
        self.h = image_size[0]
        self.w = image_size[1]
        self.scale = self.create_scale(decay_power).to(device)

    def create_scale(self, decay_power=1.0):
        freqs = self.rfft2d_freqs()
        self.freqs_shape = freqs.size() + (2,)
        scale = 1.0 / torch.max(freqs, torch.full_like(freqs, 1.0 / (max(self.w, self.h)))) ** decay_power
        return scale[None, None, ..., None]

    def rfft2d_freqs(self):
        fy = self.pytorch_fftfreq(self.h)[:, None]
        wadd = 2 if self.w % 2 == 1 else 1
        fx = self.pytorch_fftfreq(self.w)[: self.w // 2 + wadd]
        return torch.sqrt((fx * fx) + (fy * fy))

    def pytorch_fftfreq(self, v, d=1.0):
        results = torch.empty(v)
        s = (v - 1) // 2 + 1
        results[:s] = torch.arange(0, s)
        results[s:] = torch.arange(-(v // 2), 0)
        return results * (1.0 / (v * d))

    def ifft_image(self, input):
        input = input * self.scale
        input = torch.irfft(input, 2, normalized=True, signal_sizes=(self.h, self.w))
        return input / 4

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
            color_correlation_svd_sqrt = torch.Tensor([[0.26, 0.09, 0.02], #Placeholder
                                                      [0.27, 0.00, -0.05],
                                                      [0.27, -0.09, 0.03]])
        return color_correlation_svd_sqrt

    def color_correlation_normalized(self, matrix):
        color_correlation_svd_sqrt = self.get_matrix(matrix)
        max_norm_svd_sqrt = torch.max(color_correlation_svd_sqrt.norm(0))
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
        return color_correlation_normalized.T		

    def forward(self, input):
        return torch.matmul(input.permute(0,2,3,1), self.color_correlation_n).permute(0,3,1,2)


# Preprocess input after decorrelation
class TransformLayer(torch.nn.Module):

    def __init__(self, input_mean=[1,1,1], r=255, device='cpu'):
        super(TransformLayer, self).__init__()
        self.input_mean = torch.as_tensor(input_mean).to(device)
        self.input_sd = torch.as_tensor([1,1,1]).to(device)
        self.r = r

    def forward(self, input):
        input = torch.sigmoid(input) * self.r
        return input.sub(self.input_mean[None, :, None, None]).div(self.input_sd[None, :, None, None])


# Randomly scale an input
class RandomScaleLayer(torch.nn.Module):

    def __init__(self, scale_list=(1, 0.975, 1.025, 0.95, 1.05)):
        super(RandomScaleLayer, self).__init__()
        self.scale_list = scale_list

    def rescale_tensor(self, input, scale, align_corners=True):
        return torch.nn.functional.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=align_corners)

    def forward(self, input):
        n = random.randint(0, len(self.scale_list)-1)
        return self.rescale_tensor(input, scale=self.scale_list[n])
