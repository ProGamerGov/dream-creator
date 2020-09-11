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
    parser.add_argument("-output_name", type=str, default='')
    params = parser.parse_args()
    main_func(params)


def main_func(params)
    checkpoint = torch.load(params.model_file, map_location='cpu')
    save_model = copy.deepcopy(checkpoint)

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
                norm_vals[1] = [float(m) for m in params.data_sd.split(',')]
            if params.normval_format != 'ignore':
                norm_vals[2] = params.normval_format
            save_model['normalize_params'] = norm_vals
        except:
            assert params.data_mean != 'ignore', \ "'-data_mean' is required"
            assert params.data_sd != 'ignore', \ "'-data_sd' is required"
            assert params.normval_format != 'ignore', \ "'-normval_format' is required"
            save_model['normalize_params'] = [params.data_mean, params.data_sd, params.normval_format]

    torch.save(save_model, save_name)