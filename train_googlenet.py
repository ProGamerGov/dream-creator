import argparse
import torch
import torch.optim as optim

from utils.training_utils import save_model, load_dataset, reset_weights, set_seed, load_checkpoint, setup_model
from utils.inceptionv1_caffe import InceptionV1_Caffe
from utils.train_model import train_model


def main():
    parser = argparse.ArgumentParser()
    # Input options
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-model_file", type=str, default='models/pt_bvlc.pth')
    parser.add_argument("-data_mean", type=str, default='')
    parser.add_argument("-data_sd", type=str, default='')
    parser.add_argument("-base_model", choices=['bvlc', 'p365', '5h'], default='bvlc')

    # Training options
    parser.add_argument("-num_epochs", type=int, default=120)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument( "-lr", "-learning_rate", type=float, default=1e-2)
    parser.add_argument("-optimizer", choices=['sgd', 'adam'], default='sgd')
    parser.add_argument("-train_workers", type=int, default=0)
    parser.add_argument("-val_workers", type=int, default=0)
    parser.add_argument("-balance_classes", action='store_true')

    # Output options
    parser.add_argument("-save_epoch", type=int, default=10)
    parser.add_argument("-output_name", type=str, default='bvlc_out.pth')
    parser.add_argument("-individual_acc", action='store_true')
    parser.add_argument("-save_csv", action='store_true')
    parser.add_argument("-csv_dir", type=str, default='')

    # Other options
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-use_device", type=str, default='cuda:0')
    parser.add_argument("-seed", type=int, default=-1)

    # Dataset options
    parser.add_argument("-val_percent", type=float, default=0.2)

    # Model options
    parser.add_argument("-reset_weights", action='store_true')
    parser.add_argument("-delete_branches", action='store_true')
    parser.add_argument("-freeze_aux1_to", choices=['none', 'loss_conv', 'loss_fc', 'loss_classifier'], default='none')
    parser.add_argument("-freeze_aux2_to", choices=['none', 'loss_conv', 'loss_fc', 'loss_classifier'], default='none')
    parser.add_argument("-freeze_to", choices=['none', 'conv1', 'conv2', 'conv3', 'mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b'], default='mixed3b')
    params = parser.parse_args()
    main_func(params)


def main_func(params):
    assert params.data_mean != '', "-data_mean is required"
    assert params.data_sd != '', "-data_sd is required"
    params.data_mean = [float(m) for m in params.data_mean.split(',')]
    params.data_sd = [float(s) for s in params.data_sd.split(',')]

    if params.seed > -1:
        set_seed(params.seed)
    rnd_generator = torch.Generator(device='cpu') if params.seed > -1 else None

    # Setup image training data
    training_data, num_classes, class_weights = load_dataset(data_path=params.data_path, val_percent=params.val_percent, batch_size=params.batch_size, \
                                                             input_mean=params.data_mean, input_sd=params.data_sd, use_caffe=not params.not_caffe, \
                                                             train_workers=params.train_workers, val_workers=params.val_workers, balance_weights=params.balance_classes, \
                                                             rnd_generator=rnd_generator)


    # Setup model definition
    cnn, is_start_model, base_model = setup_model(params.model_file, num_classes=num_classes, base_model=params.base_model, pretrained=not params.reset_weights)

    if params.optimizer == 'sgd':
        optimizer = optim.SGD(cnn.parameters(), lr=params.lr, momentum=0.9)
    elif params.optimizer == 'adam':
        optimizer = optim.Adam(cnn.parameters(), lr=params.lr)

    lrscheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.96)

    if params.balance_classes:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(params.use_device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Maybe delete braches
    if params.delete_branches and not is_start_model:
        try:
            cnn.remove_branches()
            has_branches = False
        except:
            has_branches = True
            pass
    else:
       has_branches = True


    # Load pretrained model weights
    start_epoch = 1
    if not params.reset_weights:
        cnn, optimizer, lrscheduler, start_epoch = load_checkpoint(cnn, params.model_file, optimizer, lrscheduler, num_classes, is_start_model=is_start_model)

    if params.delete_branches and is_start_model:
        try:
            cnn.remove_branches()
            has_branches = False
        except:
            has_branches = True
            pass
    else:
       has_branches = True


    # Maybe freeze some model layers
    main_layer_list = ['conv1', 'conv2', 'conv3', 'mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
    if params.freeze_to != 'none':
        for layer in main_layer_list:
            if params.freeze_to == layer:
                break
            for param in getattr(cnn, layer).parameters():
                param.requires_grad = False
    branch_layer_list = ['loss_conv', 'loss_fc', 'loss_classifier']
    if params.freeze_aux1_to != 'none' and has_branches:
        for layer in branch_layer_list:
            if params.freeze_aux1_to == layer:
                break
            for param in getattr(getattr(cnn, 'aux1'), layer).parameters():
                param.requires_grad = False
    if params.freeze_aux2_to != 'none' and has_branches:
        for layer in branch_layer_list:
            if params.freeze_aux2_to == layer:
                break
            for param in getattr(getattr(cnn, 'aux2'), layer).parameters():
                param.requires_grad = False


    n_learnable_params = sum(param.numel() for param in cnn.parameters() if param.requires_grad)
    print('Model has ' + "{:,}".format(n_learnable_params) + ' learnable parameters\n')


    cnn = cnn.to(params.use_device)
    if 'cuda' in params.use_device:
        if params.seed > -1:
            torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


    save_info = [[params.data_mean, params.data_sd, 'BGR'], num_classes, has_branches, base_model]

    # Train model
    train_model(model=cnn, dataloaders=training_data, criterion=criterion, optimizer=optimizer, lrscheduler=lrscheduler, \
                num_epochs=params.num_epochs, start_epoch=start_epoch, save_epoch=params.save_epoch, output_name=params.output_name, \
                device=params.use_device, has_branches=has_branches, fc_only=False, num_classes=num_classes, individual_acc=params.individual_acc, \
                should_save_csv=params.save_csv, csv_path=params.csv_dir, save_info=save_info)



if __name__ == "__main__":
    main()