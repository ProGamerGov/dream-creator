import argparse
import torch
import copy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-model_file", type=str, default='')
    parser.add_argument("-num_classes", type=int, default=-1)
    parser.add_argument("-epoch", type=int, default=-1)
    parser.add_argument("-base_model", choices=['bvlc', 'p365', '5h', 'ignore'], default='ignore')
    parser.add_argument("-data_mean", type=str, default='ignore')
    parser.add_argument("-data_sd", type=str, default='ignore')
    parser.add_argument("-normval_format", choices=['bgr', 'rgb', 'ignore'], default='ignore')
    parser.add_argument("-has_branches", choices=['true', 'false', 'ignore'], default='ignore')
    parser.add_argument("-reverse_normvals", action='store_true')
    parser.add_argument("-print_vals", action='store_true')
    parser.add_argument("-output_name", type=str, default='')
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    checkpoint = torch.load(params.model_file, map_location='cpu')
    save_model = copy.deepcopy(checkpoint)
    
    if params.print_vals:
        print_model_vals(save_model)

    if params.num_classes > -1:
        save_model['num_classes'] = params.num_classes
        
    if params.base_model != 'ignore':
        save_model['base_model'] = params.base_model
        
    if params.has_branches != 'ignore':
        has_branches = True if params.has_branches == 'true' else False
        save_model['has_branches'] = has_branches

    if params.epoch != -1:
        save_model['epoch'] = params.epoch
        
    if params.data_mean != 'ignore' or params.data_sd != 'ignore' or params.normval_format != 'ignore':
        try:
            norm_vals = save_model['normalize_params']
            if params.data_mean != 'ignore':
                norm_vals[0] = [float(m) for m in params.data_mean.split(',')]
            if params.data_sd != 'ignore':
                norm_vals[1] = [float(s) for s in params.data_sd.split(',')]
            if params.normval_format != 'ignore':
                try:
                    norm_vals[2] = params.normval_format
                except:
                    norm_vals += [params.normval_format] # Add to legacy models
            save_model['normalize_params'] = norm_vals

        except:
            assert params.data_mean != 'ignore', "'-data_mean' is required"
            assert params.data_sd != 'ignore', "'-data_sd' is required"
            assert params.normval_format != 'ignore', "'-normval_format' is required"
            save_model['normalize_params'] = [params.data_mean, params.data_sd, params.normval_format]
            
        if params.reverse_normvals:
            norm_vals = save_model['normalize_params']
            norm_vals[0].reverse()
            norm_vals[1].reverse()
            save_model['normalize_params'] = norm_vals

    if params.output_name != '':
        torch.save(save_model, save_name)
    
    
    
def print_model_vals(model):
    print('Model Values')

    try:
        print('  Num classes:', model['num_classes'])
    except:
        pass
    try:
        print('  Base model:', model['base_model'])
    except:
        pass
    try:
        print('  Model epoch:', model['epoch'])
    except:
        pass
    try:
        print('  Has branches:', model['has_branches'])
    except:
        pass
    try:
        print('  Norm value format', model['normalize_params'][2])
    except:
        pass
    try:
        print('  Mean values:', model['normalize_params'][0])
    except:
        pass
    try:
        print('  Standard deviation values:', model['normalize_params'][1])
    except:
        pass
    try:
        test = model['optimizer_state_dict']
        print('  Contains saved optimizer state')
    except:
        pass
    try:
        test = model['lrscheduler_state_dict']
        print('  Contains saved learning rate scheduler state')
    except:
        pass



if __name__ == "__main__":
    main()
