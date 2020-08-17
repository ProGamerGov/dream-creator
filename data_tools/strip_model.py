import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_file", help="Path to your model", type=str, default='')
    parser.add_argument("-output_name", type=str, default='out_stripped.pth')
    parser.add_argument("-delete_branches", action='store_true')
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    checkpoint = torch.load(params.model_file, map_location='cpu')
    if type(checkpoint) == dict:
        model_keys = list(checkpoint.keys())
        if 'optimizer_state_dict' in model_keys:
            del checkpoint['optimizer_state_dict']
        if 'lrscheduler_state_dict' in model_keys:
            del checkpoint['lrscheduler_state_dict']

        if params.delete_branches:
            for layer in list(checkpoint['model_state_dict']):
                if 'aux1' in layer or 'aux2' in layer:
                    del checkpoint['model_state_dict'][layer]
            checkpoint['has_branches'] = False

    torch.save(checkpoint, params.output_name)



if __name__ == "__main__":
    main()