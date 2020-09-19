import argparse
import torch
import torch.optim as optim

from utils.training_utils import load_dataset
from utils.vis_utils import load_model, register_layer_hook
from utils.train_model import train_model


def main():
    parser = argparse.ArgumentParser()
    # Input options
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-model_file", type=str, default='models/pt_bvlc.pth')
    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-data_sd", type=str, default='')
    parser.add_argument("-base_model", choices=['bvlc', 'p365', '5h'], default='bvlc')
    parser.add_argument("-num_classes", type=int, default=32)
    parser.add_argument("-no_branches", action='store_true')

    # Training options
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-balance_classes", action='store_true')

    # Output options
    parser.add_argument("-output_name", type=str, default='activations.pt')

    # Other options
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-use_device", type=str, default='cuda:0')


    parser.add_argument("-layer", type=str, default='fc')
    params = parser.parse_args()
    main_func(params)


def main_func(params):

    cnn, norm_vals, _ = load_model(params.model_file, params.num_classes, has_branches=not params.no_branches)
    if norm_vals != None or params.data_mean == '':
        params.data_mean = norm_vals[0]
        params.data_sd = norm_vals[1]
    else:
        assert params.data_mean != '', "-data_mean is required"
        assert params.data_sd != '', "-data_sd is required"
        params.data_mean = [float(m) for m in params.data_mean.split(',')]
        params.data_sd = [float(s) for s in params.data_sd.split(',')]

 
    # Setup image training data
    eval_data, num_classes, class_weights = load_dataset(data_path=params.data_path, val_percent=0, batch_size=params.batch_size, \
                                                         input_mean=params.data_mean, input_sd=params.data_sd, use_caffe=not params.not_caffe, \
                                                         train_workers=params.num_workers, val_workers=0, balance_weights=params.balance_classes)

    eval_data = eval_data['val']
 
    cnn, norm_vals, _ = load_model(params.model_file, params.num_classes, has_branches=not params.no_branches)
    cnn.eval()
    cnn = cnn.to(params.use_device)
    if 'cuda' in params.use_device:
        torch.backends.cudnn.enabled = True

    get_activations = []
    for layer in params.layer.split(','):
       activation_catcher = AddActivations()   
       get_activations += register_layer_hook(cnn, layer, activation_catcher)  


    with torch.no_grad():
        print('Running dataset through model')
        for inputs, _ in eval_data:
            inputs = inputs.to(params.use_device)
            outputs = cnn(inputs)

    data_activ = get_activations[0].activations
    data_activ = data_activ / get_activations[0].count
    
    torch.save(data_activ, params.output_name)


class AddActivations(torch.nn.Module):
    def __init__(self):
        super(AddActivations, self).__init__()
        self.activations = None
        self.count = 0

    def forward(self, module, input, output):
        if self.activations == None:
             self.activations = torch.zeros_like(output[0])
        for b in range(output.size(0)): 
            self.activations = self.activations + output[b]
            self.count += 1



if __name__ == "__main__":
    main()