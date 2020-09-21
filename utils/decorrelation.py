import torch
import torch.nn as nn



def get_decorrelation_layers(image_size=(224,224), input_mean=[1,1,1], device='cpu', decay_power=0.75):
    spatial_mod = SpatialDecorrelationLayer(image_size, decay_power=decay_power, device=device)
    transform_mod = TransformLayer(input_mean=input_mean, device=device)
    return [spatial_mod, transform_mod]


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
